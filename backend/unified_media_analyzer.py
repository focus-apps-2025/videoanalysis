# unified_media_analyzer.py
import librosa
import numpy as np
import cv2
import pickle
import tempfile
import os
import requests
import sys
import io
import subprocess
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup
import re
import json
import shutil
import uuid
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    BartForConditionalGeneration, BartTokenizer,
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# NLLB language codes and names
INDIAN_LANGUAGE_CODES = {
    'as': 'asm_Beng', 'bn': 'ben_Beng', 'gu': 'guj_Gujr', 'hi': 'hin_Deva',
    'kn': 'kan_Knda', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'ne': 'npi_Deva',
    'or': 'ory_Orya', 'pa': 'pan_Guru', 'sa': 'san_Deva', 'ta': 'tam_Taml',
    'te': 'tel_Telu', 'ur': 'urd_Arab'
}
LANGUAGE_NAME_LOOKUP = {
    'as': 'Assamese', 'bn': 'Bengali', 'gu': 'Gujarati', 'hi': 'Hindi',
    'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali',
    'or': 'Odia', 'pa': 'Punjabi', 'sa': 'Sanskrit', 'ta': 'Tamil',
    'te': 'Telugu', 'ur': 'Urdu'
}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class UnifiedMediaAnalyzer:
    RESOLUTION_MAP = [
        (3840, 2160, "4K UHD", 85, 100),
        (2560, 1440, "1440p (QHD)", 70, 85),
        (1920, 1080, "1080p (FHD)", 55, 70),
        (1280, 720, "720p (HD)", 40, 55),
        (854, 480, "480p (SD)", 25, 40),
        (640, 360, "360p (LD)", 10, 25),
        (0, 0, "Unknown/Very Low Res", 0, 10)
    ]
    SHAKE_TOLERANCE_PX = 1.5
    NOISE_THRESHOLD_STD = 8

    def __init__(self, target_language='fr'):
        self.audio_model = None
        self.audio_scaler = None
        self.video_model = None
        self.video_scaler = None
        self.whisper_processor = None
        self.whisper_model = None
        self.summarization_model = None
        self.summarization_tokenizer = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.target_language = target_language
        self.lang_model_map = {
        
        'fr': "Helsinki-NLP/opus-mt-en-fr",
        # Use NLLB for ALL Indian languages (it supports 200+ languages including all Indian ones)
        'as': "facebook/nllb-200-distilled-600M",
        'bn': "facebook/nllb-200-distilled-600M",
        'gu': "facebook/nllb-200-distilled-600M",
        'hi': "facebook/nllb-200-distilled-600M",
        'kn': "facebook/nllb-200-distilled-600M",
        'ml': "facebook/nllb-200-distilled-600M",
        'mr': "facebook/nllb-200-distilled-600M",
        'ne': "facebook/nllb-200-distilled-600M",
        'or': "facebook/nllb-200-distilled-600M",
        'pa': "facebook/nllb-200-distilled-600M",
        'sa': "facebook/nllb-200-distilled-600M",
        'ta': "facebook/nllb-200-distilled-600M",
        'te': "facebook/nllb-200-distilled-600M",  # ✅ Now correct!
        'ur': "facebook/nllb-200-distilled-600M",

        }
        
        self.models_loaded = False
        self._check_ffmpeg_path()

    def _check_ffmpeg_path(self):
        if shutil.which("ffmpeg") is None:
            print("\n" + "="*80)
            print("WARNING: FFmpeg not found in system PATH.")
            print("Please ensure FFmpeg is installed and its 'bin' directory is added to your system's PATH.")
            print("="*80 + "\n")

    def _get_resolution_info(self, width, height):
        sorted_map = sorted(self.RESOLUTION_MAP, key=lambda x: x[0]*x[1], reverse=True)
        for res_w, res_h, label, min_score, max_score in sorted_map:
            if res_w == 0 and res_h == 0:
                continue
            if (width >= res_w and height >= res_h) or (height >= res_w and width >= res_h):
                return label, min_score, max_score
        return sorted_map[-1][2], sorted_map[-1][3], sorted_map[-1][4]

    def extract_citnow_metadata(self, url):
        "Extract CitNow page metadata with robust handling of tables, vehicle/vin/registration, and dealership info.Supports modern BMW/MINI/MG/other templates."
        print("🌐 Extracting CitNow page metadata...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text(separator='\n')

            metadata = {
                'page_url': url,
                'extraction_timestamp': datetime.now().isoformat(),
                # new
                'brand': None,
                'dealership': None,
                'vehicle': None,
                'registration': None,
                'vin': None,
                # as before
                'service_advisor': None,
                'email': None,
                'phone': None,
            }

            # Brand detection by title/img
            title = soup.title.string if soup.title else ""
            if 'BMW' in title or soup.find('img', src=lambda x: x and 'bmw' in x.lower()):
                metadata['brand'] = 'BMW'
            elif 'MINI' in title or soup.find('img', src=lambda x: x and 'mini' in x.lower()):
                metadata['brand'] = 'MINI'
            elif 'MG' in title or soup.find('img', src=lambda x: x and 'mg' in x.lower()):
                metadata['brand'] = 'MG'

            # Dealership extraction with improved logic
            metadata['dealership'] = self._extract_dealership(soup, page_text)

            # ---- TABLE-BASED extraction preferred (BMW/MINI templates and many others) ----
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).replace(':', '').strip().lower()
                        value = cells[1].get_text(strip=True)

                        # VIN: 17 chars, A-HJ-NPR-Z and numbers (except I/O/Q)
                        if any(kw in label for kw in ['vin', 'chassis']):
                            if re.fullmatch(r'[A-HJ-NPR-Z0-9]{17}', value):
                                metadata['vin'] = value
                        # Registration: typical plate pattern (contains letters and numbers)
                        elif any(kw in label for kw in ['reg', 'registration', 'plate', 'license']):
                            metadata['registration'] = value
                        # Vehicle model/info
                        elif 'vehicle' in label:
                            metadata['vehicle'] = value

                        # Service advisor/technician
                        elif any(kw in label for kw in ['advisor', 'technician', 'service', 'presenter']):
                            metadata['service_advisor'] = value

                        # Contact details
                        elif 'email' in label:
                            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', value)
                            if email_match:
                                metadata['email'] = email_match.group(1)
                        elif 'phone' in label or 'mobile' in label:
                            phone_match = re.search(r'(\+?\d[\d\s-]{8,})', value)
                            if phone_match:
                                metadata['phone'] = phone_match.group(1)

            # Fallback: regex search in full plain text if table parsing failed

            # Service advisor
            if not metadata['service_advisor']:
                advisor_match = re.search(r'(?:Technician|Service Advisor)[\s:]*([A-Za-z]+(?:,?\s+[A-Za-z]+)*)', page_text, re.IGNORECASE)
                if advisor_match:
                    metadata['service_advisor'] = advisor_match.group(1).strip()

            # Email
            if not metadata['email']:
                email_match = re.search(r'Email[\s:]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', page_text, re.IGNORECASE)
                if email_match:
                    metadata['email'] = email_match.group(1).strip()

            # Phone
            if not metadata['phone']:
                phone_match = re.search(r'Phone[\s:]*(\+?\d[\d\s-]{8,})', page_text, re.IGNORECASE)
                if phone_match:
                    metadata['phone'] = phone_match.group(1).strip()

            # If VIN or registration still missing, try smarter regexes in text:
            if not metadata['vin']:
                vin_match = re.search(r'\b([A-HJ-NPR-Z0-9]{17})\b', page_text)
                if vin_match:
                    metadata['vin'] = vin_match.group(1).strip()
            if not metadata['registration']:
                reg_match = re.search(r'(?:Registration|Reg\.?|Plate)\s*[:\-]*\s*([A-Za-z0-9\-\s]{5,15})', page_text)
                if reg_match:
                    maybe_reg = reg_match.group(1).strip()
                    if len(maybe_reg) <= 15:
                        metadata['registration'] = maybe_reg
            

            video_url = self._extract_video_url_from_page(soup, page_text)
            if video_url:
                metadata['video_url'] = video_url

            print("✅ Metadata extracted successfully!")
            return metadata
        except Exception as e:
            print(f"⚠️ Error extracting metadata: {e}")
            return {
                'page_url': url,
                'error': str(e),
                'extraction_timestamp': datetime.now().isoformat()
        }

    def _extract_dealership(self, soup, page_text):
        candidates = set()
        keywords = ["Private Limited", "Pvt Ltd", "Motors", "Cars", "Dealer", "Automotive", "Showroom", "LLP"]
        for tag in ['h1','h2','h3']:
            for el in soup.find_all(tag):
                t = el.get_text(separator=' ', strip=True)
                if any(kw in t for kw in keywords) and 8 < len(t) < 80:
                    candidates.add(t.split('\n')[0].strip())
        m = re.search(r"\bfrom\s+([A-Z][A-Za-z0-9&.,\- ]{3,}(?:Pvt\.?\s?Ltd|Private Limited|Motors|Cars|Automotive|Showroom|LLP|Limited|Services|Ahmedabad|Chennai|Bangalore|Delhi|Mumbai)?)", page_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if len(val) < 80:
                candidates.add(val)
        for line in page_text.split('\n'):
            s = line.strip()
            if any(kw in s for kw in keywords) and 6 < len(s) < 80 and not any(w in s for w in ["browser", "JavaScript", "support", "disable", "presentation"]):
                candidates.add(s)
        clean_candidates = []
        for c in candidates:
            c = re.split(r"Vehicle:|Presenter:|Service Advisor|Phone|Email|Call|If you would like|browser", c)[0].strip()
            if len(c) > 6:
                clean_candidates.append(c)
        if clean_candidates:
            def keyword_count(s):
                return sum(1 for kw in keywords if kw in s)
            sorted_clean = sorted(clean_candidates, key=lambda x: (-keyword_count(x), len(x)))
            return sorted_clean[0]
        return None

    

    def _extract_video_url_from_page(self, soup, page_text):
        try:
            video_elem = soup.find('video', {'src': True})
            if video_elem:
                return self._clean_url(video_elem['src'])
            video_elem = soup.find('video')
            if video_elem:
                source = video_elem.find('source', {'src': True})
                if source:
                    return self._clean_url(source['src'])
            iframe = soup.find('iframe', {'src': True})
            if iframe and 'video' in iframe['src']:
                return self._clean_url(iframe['src'])
            mp4_pattern = r'(https?://[^\s"]+\.mp4)'
            mp4_matches = re.findall(mp4_pattern, page_text)
            if mp4_matches:
                return self._clean_url(mp4_matches[0])
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    video_patterns = [
                        r'videoUrl["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                        r'src["\']?\s*[:=]\s*["\']([^"\']+\.mp4)["\']',
                        r'file["\']?\s*[:=]\s*["\']([^"\']+\.mp4)["\']',
                        r'"url"\s*:\s*"([^"]+\.mp4)"',
                        r'"video"\s*:\s*"([^"]+\.mp4)"'
                    ]
                    for pattern in video_patterns:
                        match = re.search(pattern, script.string)
                        if match:
                            video_url = match.group(1)
                            video_url = self._clean_url(video_url)
                            if not video_url.startswith('http'):
                                video_url = 'https://southasia.citnow.com/' + video_url.lstrip('/')
                            return video_url
            return None
        except Exception as e:
            print(f"⚠️ Error extracting video URL: {e}")
            return None

    def _clean_url(self, url):
        if not url:
            return None
        url = url.replace('\\/\\/', '//').replace('\\/', '/').replace('\\', '')
        if url.startswith('//'):
            url = 'https:' + url
        elif not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url

    def download_citnow_video(self, url):
        print("📥 Attempting to download CitNow video...")
        metadata = self.extract_citnow_metadata(url)
        video_url = metadata.get('video_url')
        if video_url:
            video_url = self._clean_url(video_url)
            print(f"🎥 Found video URL: {video_url[:50]}...")
            try:
                return self._download_from_url(video_url)
            except Exception as e:
                print(f"⚠️ Direct download failed: {e}")
        try:
            import yt_dlp
            temp_dir = tempfile.gettempdir()
            unique_filename = f"citnow_{uuid.uuid4().hex}"
            output_template = os.path.join(temp_dir, f"{unique_filename}.%(ext)s")
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'user_agent': 'Mozilla/5.0'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("⬇️ Trying yt-dlp download...")
                info = ydl.extract_info(url, download=False)
                ext = info.get('ext', 'mp4')
                video_path = os.path.join(temp_dir, f"{unique_filename}.{ext}")
                ydl.download([url])
                if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                    print("✅ Video downloaded successfully with yt-dlp!")
                    return video_path
                else:
                    raise Exception("yt-dlp downloaded file not found or invalid size.")
        except Exception as e:
            print(f"⚠️ yt-dlp download failed: {e}")
        video_id_match = re.search(r'/([a-zA-Z0-9]+)$', url)
        if video_id_match:
            video_id = video_id_match.group(1)
            possible_urls = [
                f'https://southasia.citnow.com/videos/{video_id}.mp4',
                f'https://southasia.citnow.com/video/{video_id}/video.mp4',
                f'https://cdn.citnow.com/{video_id}.mp4',
                f'https://lts.in.prod.citnow.com/cin-southasia/{video_id}/output-1200k.mp4'
            ]
            for test_url in possible_urls:
                try:
                    print(f"🔍 Trying fallback URL: {test_url}")
                    return self._download_from_url(test_url)
                except Exception:
                    continue
        raise Exception("Could not download video from CitNow using any method.")

    def load_pretrained_models(self):
        if self.models_loaded:
            return
        print("🔄 Loading all pre-trained models...")
        whisper_model_name = "openai/whisper-medium"
        print(f"🎙️ Loading Whisper model: {whisper_model_name}")
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        print("📝 Loading BART summarization model: facebook/bart-large-cnn")
        self.summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        model_name = self.lang_model_map.get(self.target_language, self.lang_model_map['fr'])
        print(f"🌍 Loading translation model: {model_name}")
        
        # ALWAYS use Auto* for NLLB; MarianMT only for non-NLLB (like 'fr')
        if model_name == "facebook/nllb-200-distilled-600M":
            self.translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            self.translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
        
        self.models_loaded = True
        print(f"✅ All pre-trained models loaded! (Target Language: {self.target_language.upper()})")
    def extract_audio_from_video(self, video_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio_path = temp_audio_file.name
            try:
                cmd = ['ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y']
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                    print("✅ Audio extracted successfully")
                    return audio_path
                else:
                    print(f"⚠️ FFmpeg output (stderr):\n{result.stderr}")
                    raise Exception("Audio file too small or invalid after extraction")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ FFmpeg command failed with error code {e.returncode}: {e.stderr}")
                if os.path.exists(audio_path): os.unlink(audio_path)
                raise Exception(f"FFmpeg extraction failed: {e.stderr}")
            except Exception as e:
                print(f"⚠️ Audio extraction failed: {e}")
                if os.path.exists(audio_path): os.unlink(audio_path)
                raise Exception(f"Audio extraction failed: {e}")

    def analyze_audio_quality(self, audio_path):
        if self.audio_model is None:
            return self._default_audio_analysis(audio_path)
        try:
            features = self.extract_audio_features(audio_path)
            features = np.nan_to_num(features).reshape(1, -1)
            if self.audio_scaler:
                features = self.audio_scaler.transform(features)
            prediction = self.audio_model.predict(features)[0]
            probability = self.audio_model.predict_proba(features)[0]
            return {
                'prediction': 'Clear' if prediction == 1 else 'Noisy',
                'confidence': float(max(probability)),
                'score': float(probability[1])
            }
        except Exception as e:
            return {'error': f'Audio analysis failed: {e}'}

    def analyze_video_quality(self, video_path):
        if self.video_model is None:
            return self._default_video_analysis(video_path)
        try:
            features = self.extract_video_features(video_path)
            features = np.nan_to_num(features).reshape(1, -1)
            if self.video_scaler:
                features = self.video_scaler.transform(features)
            score = self.video_model.predict(features)[0]
            score = max(0, min(100, score))
            return {
                'quality_score': float(score),
                'quality_label': self._get_quality_label(score)
            }
        except Exception as e:
            return {'error': f'Video analysis failed: {e}'}

    # ✅ FULLY FIXED: Load full audio + increased max_length
    '''def transcribe_audio(self, audio_path, transcription_language=None):
        if self.whisper_processor is None or self.whisper_model is None:
            self.load_pretrained_models()
        audio, sr = librosa.load(audio_path, sr=16000)
        segment_len_sec = 28  # keep under 30 to be safe
        samples_per_window = int(segment_len_sec * sr)
        transcript = ""
        total_secs = len(audio) / sr
        print(f"🔊 Loaded full audio: {total_secs:.1f} seconds")
        for i, start in enumerate(range(0, len(audio), samples_per_window)):
            chunk = audio[start:start+samples_per_window]
            input_features = self.whisper_processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
            if not transcription_language or transcription_language == 'auto':
                predicted_ids = self.whisper_model.generate(input_features, max_length=448)
            else:
                forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language=transcription_language, task="transcribe")
                predicted_ids = self.whisper_model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids, max_length=448
                )
            txt = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            transcript += " " + txt
        return transcript.strip()
 '''
    def transcribe_audio(self, audio_path, transcription_language=None):
        if self.whisper_processor is None or self.whisper_model is None:
            self.load_pretrained_models()
        audio, sr = librosa.load(audio_path, sr=16000)
        segment_len_sec = 28
        samples_per_window = int(segment_len_sec * sr)
        transcript = ""
        total_secs = len(audio) / sr
        print(f"🔊 Loaded full audio: {total_secs:.1f} seconds")
        for i, start in enumerate(range(0, len(audio), samples_per_window)):
            chunk = audio[start:start+samples_per_window]
            input_features = self.whisper_processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
            if not transcription_language or transcription_language == 'auto':
                predicted_ids = self.whisper_model.generate(
                    input_features=input_features, max_length=448
                )
            else:
                forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language=transcription_language, task="transcribe")
                predicted_ids = self.whisper_model.generate(
                    input_features=input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448
                )
            txt = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            transcript += " " + txt
        return transcript.strip()

    def summarize_text(self, text):
        if not self.summarization_model:
            self.load_pretrained_models()
        if len(text.strip()) < 50:
            return "Text too short for meaningful summary"
        try:
            inputs = self.summarization_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = self.summarization_model.generate(
                inputs["input_ids"],
                max_length=250,
                min_length=60,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = self.summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return f"Summarization error: {e}"

    def translate_text(self, text):
        if not self.translation_model:
            self.load_pretrained_models()
        try:
            if len(text.strip()) < 10:
                return "Input text too short to translate"

            # Check if using NLLB
            model_name = self.lang_model_map.get(self.target_language, self.lang_model_map['fr'])
            if model_name == "facebook/nllb-200-distilled-600M":
                tokenizer = self.translation_tokenizer
                model = self.translation_model
                flores_code = INDIAN_LANGUAGE_CODES.get(self.target_language, "eng_Latn")
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(flores_code)
                if not isinstance(forced_bos_token_id, int):
                    forced_bos_token_id = forced_bos_token_id[0]
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512
                )
                return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            else:
                # MarianMT path (e.g., French)
                inputs = self.translation_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = self.translation_model.generate(**inputs, max_length=512)
                return self.translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            return f"Translation error: {e}"
    def process_video(self, video_input, transcription_language=None, target_language_short=None):
        if target_language_short:
            self.target_language = target_language_short
        print("\n" + "="*60)
        print("🚀 ENHANCED UNIFIED VIDEO ANALYSIS PIPELINE")
        print("="*60)
        results = {
            'input_source': video_input,
            'processing_timestamp': datetime.now().isoformat(),
            'processing_steps': [],
            'target_language': self.target_language
        }
        temp_files_to_clean = []
        try:
            if isinstance(video_input, str) and 'citnow.com' in video_input:
                print("\n📊 EXTRACTING CITNOW METADATA")
                print("-" * 40)
                metadata = self.extract_citnow_metadata(video_input)
                results['citnow_metadata'] = metadata
                results['processing_steps'].append('metadata_extraction')
                print(f"🏢 Dealership: {metadata.get('dealership', 'Not found')}")
                print(f"🚗 Vehicle: {metadata.get('vehicle', 'Not found')}")
                print(f"👤 Service Advisor: {metadata.get('service_advisor', 'Not found')}")
                print(f"📧 Email: {metadata.get('email', 'Not found')}")
                print(f"📞 Phone: {metadata.get('phone', 'Not found')}")

                print("\n📥 DOWNLOADING VIDEO")
                print("-" * 40)
                video_path = self.download_citnow_video(video_input)
                temp_files_to_clean.append(video_path)
            else:
                video_path = self._handle_input(video_input)
                if video_path != video_input:
                    temp_files_to_clean.append(video_path)

            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")

            print("\n🔊 EXTRACTING AUDIO")
            print("-" * 40)
            audio_path = self.extract_audio_from_video(video_path)
            temp_files_to_clean.append(audio_path)
            results['processing_steps'].append('audio_extraction')

            print("\n🎥 ANALYZING VIDEO QUALITY")
            print("-" * 40)
            results['video_analysis'] = self.analyze_video_quality(video_path)
            results['processing_steps'].append('video_quality_analysis')
            print(f"Quality: {results['video_analysis']['quality_label']} ({results['video_analysis']['quality_score']:.1f}/100)")

            print("\n🔊 ANALYZING AUDIO QUALITY")
            print("-" * 40)
            results['audio_analysis'] = self.analyze_audio_quality(audio_path)
            results['processing_steps'].append('audio_quality_analysis')
            print(f"Clarity: {results['audio_analysis']['prediction']}")

            print("\n💬 CONVERTING SPEECH TO TEXT")
            print("-" * 40)
            transcription = self.transcribe_audio(audio_path, transcription_language=transcription_language)
            results['transcription'] = {
                'text': transcription,
                'length': len(transcription),
                'language': transcription_language or 'en'
            }
            results['processing_steps'].append('speech_to_text')
            print(f"✅ Full transcription completed ({len(transcription)} characters)")

            print("\n📝 GENERATING SUMMARY")
            print("-" * 40)
            summary = self.summarize_text(transcription)
            results['summarization'] = {
                'summary': summary,
                'length': len(summary),
                'reduction_ratio': f"{((1 - len(summary)/len(transcription)) * 100):.1f}%" if len(transcription) > 0 else "N/A"
            }
            results['processing_steps'].append('text_summarization')
            print(f"Summary generated ({results['summarization']['reduction_ratio']} reduction)")

            print(f"\n🌍 TRANSLATING TO {self.target_language.upper()}")
            print("-" * 40)
            translation = self.translate_text(transcription)
            results['translation'] = {
                'translated_text': translation,
                'target_language': self.target_language,
                'length': len(translation)
            }
            results['processing_steps'].append('translation')
            print(f"Translation completed ({len(translation)} characters)")

        except Exception as pipeline_e:
            print(f"\n❌ Pipeline stopped due to an error: {pipeline_e}")
            results['error_message'] = str(pipeline_e)
        finally:
            for temp_file in temp_files_to_clean:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        print(f"⚠️ Could not delete temp file {temp_file}: {e}")
        print("\n✅ ALL ANALYSES COMPLETED!")
        return results

    def _handle_input(self, input_source):
        if isinstance(input_source, str):
            if input_source.startswith(('http://', 'https://')):
                if "youtube.com" in input_source or "youtu.be" in input_source:
                    print("⬇️ Downloading YouTube video...")
                    return self._download_youtube_video(input_source)
                else:
                    return self._download_from_url(input_source)
            elif os.path.exists(input_source):
                return input_source
        raise ValueError("Input must be a valid file path or URL")

    def _download_youtube_video(self, url):
        try:
            import yt_dlp
            temp_dir = tempfile.gettempdir()
            unique_filename = f"youtube_{uuid.uuid4().hex}"
            output_template = os.path.join(temp_dir, f"{unique_filename}.%(ext)s")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                ext = info.get('ext', 'mp4')
                video_path = os.path.join(temp_dir, f"{unique_filename}.{ext}")
                ydl.download([url])
                if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                    return video_path
                else:
                    raise Exception("Downloaded YouTube file not found or invalid.")
        except Exception as e:
            raise Exception(f"YouTube download failed: {e}")

    def _download_from_url(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            ext = self._get_file_extension(url, response.headers.get('content-type', ''))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            with open(temp_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return temp_file.name
        except Exception as e:
            raise Exception(f"URL download failed: {e}")

    def _get_file_extension(self, url, content_type):
        parsed = urlparse(url)
        path_ext = os.path.splitext(parsed.path)[1]
        if path_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return path_ext
        if 'mp4' in content_type:
            return '.mp4'
        elif 'avi' in content_type:
            return '.avi'
        elif 'quicktime' in content_type:
            return '.mov'
        elif 'webm' in content_type:
            return '.webm'
        return '.mp4'

    def extract_audio_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            if len(y) == 0:
                return np.zeros(19)
            features = []
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            rmse = librosa.feature.rms(y=y)
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            features.extend(np.mean(mfccs, axis=1))
            features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])
            features.extend([np.mean(rmse), np.std(rmse)])
            return np.array(features)
        except Exception as e:
            return np.zeros(19)

    def extract_video_features(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            features = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_every = max(1, total_frames // 10)
            while frame_count < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % sample_every == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    features.extend([sharpness, brightness, contrast])
                frame_count += 1
            cap.release()
            return np.pad(np.array(features), (0, 30 - len(features)), 'constant')
        except Exception as e:
            return np.zeros(30)

    def _default_audio_analysis(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            if len(y) == 0:
                return {'prediction': 'Unknown', 'confidence': 0.0, 'score': 0.0, 'note': 'Empty audio'}
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            S = np.abs(librosa.stft(y))
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=S))
            clarity_score = np.clip(rms * 5, 0, 1.0)
            noise_penalty_zcr = np.clip(1 - (zcr * 3), 0, 1.0)
            noise_penalty_flatness = np.clip(1 - (spectral_flatness * 2), 0, 1.0)
            final_score = (clarity_score * 0.5 + noise_penalty_zcr * 0.3 + noise_penalty_flatness * 0.2) * 100
            final_score = np.clip(final_score, 0, 100)
            return {
                'prediction': 'Clear' if final_score > 60 else 'Noisy',
                'confidence': round(float(final_score / 100), 2),
                'score': round(float(final_score), 1),
                'note': 'Improved scoring based on loudness, noise (ZCR), and spectral flatness.'
            }
        except Exception as e:
            return {'prediction': 'Unknown', 'confidence': 0.5, 'score': 50, 'error': f'Audio analysis failed: {e}'}

    def _calculate_shake(self, video_path, sample_idxs):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return 50.0
            lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            total_displacement = 0
            frame_pairs = 0
            prev_gray = None
            for i, idx in enumerate(sample_idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                    if p0 is not None:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                        if p1 is not None and st is not None:
                            good_new = p1[st==1]
                            good_old = p0[st==1]
                            if len(good_new) > 5:
                                displacement = np.mean(np.linalg.norm(good_new - good_old, axis=1))
                                total_displacement += displacement
                                frame_pairs += 1
                prev_gray = gray
            cap.release()
            if frame_pairs == 0: return 50.0
            avg_shake_px = total_displacement / frame_pairs
            shake_score = np.clip(100 - (avg_shake_px / self.SHAKE_TOLERANCE_PX) * 50, 0, 100)
            return shake_score
        except Exception as e:
            return 50.0

    def _calculate_noise(self, video_path, sample_idxs):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return 50.0
            noise_estimates = []
            for idx in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                if laplacian.size > 0:
                    mad = np.median(np.abs(laplacian - np.median(laplacian)))
                    if mad > 0:
                        noise_estimates.append(mad)
            cap.release()
            if not noise_estimates: return 50.0
            avg_noise_mad = np.mean(noise_estimates)
            noise_score = np.clip(100 - (avg_noise_mad / self.NOISE_THRESHOLD_STD) * 50, 0, 100)
            return noise_score
        except Exception as e:
            return 50.0

    def _default_video_analysis(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'quality_score': 0.0, 'quality_label': 'Error', 'note': 'Could not open video file.'}
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0 or fps == 0 or width == 0 or height == 0:
                return {'quality_score': 10.0, 'quality_label': 'Very Poor', 'note': 'Invalid video.'}
            res_label, min_res_score, max_res_score = self._get_resolution_info(width, height)
            num_frames_to_sample = min(60, frame_count)
            sample_idxs = np.linspace(0, frame_count - 1, num_frames_to_sample, dtype=int)
            sharpness_vals, brightness_vals, contrast_vals = [], [], []
            prev_frame_gray = None
            frozen_frame_count = 0
            grabbed = 0
            for idx in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame_gray is not None:
                    diff = cv2.absdiff(gray, prev_frame_gray)
                    if np.mean(diff) < 1.0:
                        frozen_frame_count += 1
                prev_frame_gray = gray
                sharpness_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness_vals.append(np.mean(gray))
                contrast_vals.append(np.std(gray))
                grabbed += 1
            cap.release()
            if grabbed < 5:
                return {'quality_score': 10.0, 'quality_label': 'Very Poor', 'note': 'Too few frames.'}
            def rm_outliers(a):
                a = np.array(a)
                if len(a) < 3: return a
                q1, q3 = np.percentile(a, [25, 75])
                iqr = q3 - q1
                mask = (a >= q1 - 1.5*iqr) & (a <= q3 + 1.5*iqr)
                return a[mask] if np.sum(mask) >= 3 else a
            sharpness = np.mean(rm_outliers(sharpness_vals))
            brightness = np.mean(rm_outliers(brightness_vals))
            contrast = np.mean(rm_outliers(contrast_vals))
            sharpness_norm = np.clip((sharpness - 20) / (150 - 20) * 100, 0, 100)
            brightness_norm = np.clip(100 - abs(brightness - 130) / 80 * 100, 0, 100)
            contrast_norm = np.clip((contrast - 10) / (60 - 10) * 100, 0, 100)
            shake_score = self._calculate_shake(video_path, sample_idxs)
            noise_score = self._calculate_noise(video_path, sample_idxs)
            perceived_visual_score = (sharpness_norm * 0.25 + brightness_norm * 0.15 + contrast_norm * 0.10 + shake_score * 0.30 + noise_score * 0.20)
            perceived_visual_score = np.clip(perceived_visual_score, 0, 100)
            range_size = max_res_score - min_res_score
            final_score = min_res_score + (perceived_visual_score / 100) * range_size
            final_score = np.clip(final_score, 0, 100)
            note_parts = [f"Resolution: {res_label} ({width}x{height} pixels)."]
            perceived_issues = []
            if sharpness_norm < 40: perceived_issues.append("blurry/soft footage")
            if brightness_norm < 40: perceived_issues.append("too dark/bright")
            if contrast_norm < 40: perceived_issues.append("low contrast")
            if shake_score < 40: perceived_issues.append("excessive camera shake")
            if noise_score < 40: perceived_issues.append("significant visual noise")
            if frozen_frame_count > (num_frames_to_sample * 0.3):
                perceived_issues.append("significant frozen/static frames detected")
            if perceived_issues:
                note_parts.append("Perceived issues: " + "; ".join(perceived_issues) + ".")
            return {
                'quality_score': round(float(final_score), 1),
                'quality_label': self._get_quality_label(final_score),
                'note': " ".join(note_parts)
            }
        except Exception as e:
            return {'quality_score': 10.0, 'quality_label': 'Very Poor', 'note': f"Video analysis failed: {e}"}

    def _get_quality_label(self, score):
        if score >= 85: return "Excellent"
        elif score >= 70: return "Good"
        elif score >= 50: return "Fair"
        elif score >= 25: return "Poor"
        else: return "Very Poor"

    def save_models(self, filepath="unified_models.pkl"):
        model_data = {
            'audio_model': self.audio_model,
            'audio_scaler': self.audio_scaler,
            'video_model': self.video_model,
            'video_scaler': self.video_scaler,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Models saved to {filepath}")

    def load_models(self, filepath="unified_models.pkl"):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.audio_model = model_data['audio_model']
            self.audio_scaler = model_data['audio_scaler']
            self.video_model = model_data['video_model']
            self.video_scaler = model_data['video_scaler']
            print(f"✅ Models loaded from {filepath}")
        except FileNotFoundError:
            print("⚠️ No saved custom ML models found. Using default analysis rules.")
        except Exception as e:
            print(f"⚠️ Error loading custom ML models: {e}. Using default analysis rules.")

    def generate_comprehensive_report(self, results):
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE VIDEO ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {results['processing_timestamp']}")
        report.append(f"Target Language: {results['target_language'].upper()}")
        report.append(f"Processing Steps: {', '.join(results['processing_steps'])}")
        report.append("")
        if 'citnow_metadata' in results:
            report.append("CITNOW SERVICE INFORMATION")
            report.append("-" * 60)
            meta = results['citnow_metadata']
            report.append(f"Dealership: {meta.get('dealership', 'N/A')}")
            report.append(f"Vehicle Registration: {meta.get('vehicle', 'N/A')}")
            report.append(f"Service Advisor: {meta.get('service_advisor', 'N/A')}")
            report.append(f"Contact Email: {meta.get('email', 'N/A')}")
            report.append(f"Contact Phone: {meta.get('phone', 'N/A')}")
            report.append(f"Source URL: {meta.get('page_url', 'N/A')}")
            report.append("")
        if 'video_analysis' in results:
            report.append("VIDEO QUALITY ANALYSIS")
            report.append("-" * 60)
            video = results['video_analysis']
            report.append(f"Quality Score: {video.get('quality_score', 0):.1f}/100")
            report.append(f"Quality Label: {video.get('quality_label', 'N/A')}")
            if 'note' in video:
                report.append(f"Note: {video['note']}")
            report.append("")
        if 'audio_analysis' in results:
            report.append("AUDIO ANALYSIS")
            report.append("-" * 60)
            audio = results['audio_analysis']
            report.append(f"Clarity: {audio.get('prediction', 'N/A')}")
            report.append(f"Confidence: {audio.get('confidence', 0):.2%}")
            if 'note' in audio:
                report.append(f"Note: {audio['note']}")
            report.append("")
        if 'transcription' in results:
            report.append("FULL TRANSCRIPTION")
            report.append("-" * 60)
            report.append(results['transcription']['text'])
            report.append("")
            report.append(f"Length: {results['transcription']['length']} characters")
            report.append("")
        if 'summarization' in results:
            report.append("SUMMARY")
            report.append("-" * 60)
            report.append(results['summarization']['summary'])
            report.append("")
            report.append(f"Reduction: {results['summarization']['reduction_ratio']}")
            report.append("")
        if 'translation' in results:
            report.append(f"TRANSLATION ({results['target_language'].upper()})")
            report.append("-" * 60)
            report.append(results['translation']['translated_text'])
            report.append("")
            report.append(f"Length: {results['translation']['length']} characters")
            report.append("")
        if 'error_message' in results:
            report.append("ERROR DETAILS")
            report.append("-" * 60)
            report.append(f"Pipeline encountered an error: {results['error_message']}")
            report.append("")
        return "\n".join(report)
def main():
    # Keep the analyzer and its loaded models in memory
    analyzer = None
    
    while True: # ✅ Main loop to keep the program running
        citnow_url = input("Enter CitNow video URL (or type 'exit' to quit): ").strip()
        
        # Add a way to exit the loop
        if citnow_url.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break # Exit the while loop

        print("\nAvailable Indian languages for both transcription and translation:")
        for code, name in LANGUAGE_NAME_LOOKUP.items():
            print(f"  {code} - {name}")
            
        transcription_language = input("Enter SPOKEN language code (e.g. hi, ta, 'auto'): ").strip().lower()
        target_language_short = input("Enter TARGET language code for translation (e.g. hi, ta): ").strip().lower()

        # --- Model Loading Optimization ---
        # Only create and load models on the first run
        if analyzer is None:
            print("\nFirst run: Initializing and loading models...")
            analyzer = UnifiedMediaAnalyzer(target_language=target_language_short)
            # You can optionally load pre-trained custom models here if they exist
            # analyzer.load_models("trained_models.pkl") 
        else:
            # For subsequent runs, just update the target language
            analyzer.target_language = target_language_short
            print("\nReady for next analysis...")

        # --- Process the Video ---
        results = analyzer.process_video(
            citnow_url,
            transcription_language=transcription_language,
            target_language_short=target_language_short
        )
        
        # --- Save a unique report for each video ---
        report = analyzer.generate_comprehensive_report(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sanitize the URL to create a valid filename
        safe_url_part = re.sub(r'[^a-zA-Z0-9]', '_', citnow_url.split('/')[-1])
        
        report_filename_txt = f"analysis_{safe_url_part}_{timestamp}.txt"
        report_filename_json = f"analysis_{safe_url_part}_{timestamp}.json"

        with open(report_filename_txt, 'w', encoding='utf-8') as f:
            f.write(report)
        with open(report_filename_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ Analysis complete! Reports saved as '{report_filename_txt}' and '{report_filename_json}'.")
        print("-" * 80) # Separator for the next run


if __name__ == "__main__":
    main()