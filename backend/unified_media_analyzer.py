import librosa
import numpy as np
import cv2
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
    BartForConditionalGeneration,
    BartTokenizer,
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
from pydub import AudioSegment
import warnings
import numpy as np
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# NLLB language codes and names
INDIAN_LANGUAGE_CODES = {
    "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "ne": "npi_Deva",
    "or": "ory_Orya", "pa": "pan_Guru", "sa": "san_Deva", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab",
     "en": "eng_Latn", # Keep English code for NLLB
}
LANGUAGE_NAME_LOOKUP = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "en": "English", # Added English here too
}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

class UnifiedMediaAnalyzer:
    RESOLUTION_MAP = [
        (3840, 2160, "4K UHD", 85, 100),
        (2560, 1440, "1440p (QHD)", 70, 85),
        (1920, 1080, "1080p (FHD)", 55, 70),
        (1280, 720, "720p (HD)", 40, 55),
        (854, 480, "480p (SD)", 25, 40),
        (640, 360, "360p (LD)", 10, 25),
        (0, 0, "Unknown/Very Low Res", 0, 10),
    ]
    SHAKE_TOLERANCE_PX = 1.5
    NOISE_THRESHOLD_STD = 8

    def __init__(self, target_language="en"):
        self.audio_model = None
        self.audio_scaler = None
        self.video_model = None
        self.video_scaler = None
        self.whisper_processor = None
        self.whisper_model = None
        self.summarization_model = None
        self.summarization_tokenizer = None
        
        # self.target_language is now only for initial default or for overall
        # tracking outside of actual translation model loading.
        # It's better to pass the target_language explicitly to translate_text.
        self.target_language = target_language

        # NEW: Cache for translation models, keyed by target language code
        self._translation_models = {} # Stores {lang_code: (model, tokenizer)}

        # OLD: Remove these, they are managed by _translation_models now
        # self.translation_model = None
        # self.translation_tokenizer = None
        # self.current_loaded_translation_lang = None # This is no longer needed

        # Updated language model map to primarily use NLLB for Indian languages + English
        self.lang_model_map = {
            "en": NLLB_MODEL,   
            "as": NLLB_MODEL,
            "bn": NLLB_MODEL,
            "gu": NLLB_MODEL,
            "hi": NLLB_MODEL,
            "kn": NLLB_MODEL,
            "ml": NLLB_MODEL,
            "mr": NLLB_MODEL,
            "ne": NLLB_MODEL,
            "or": NLLB_MODEL,
            "pa": NLLB_MODEL,
            "sa": NLLB_MODEL,
            "ta": NLLB_MODEL,
            "te": NLLB_MODEL,
            "ur": NLLB_MODEL,
        }

        self.faster_whisper_model = None
        self._check_ffmpeg_path()

    def _check_ffmpeg_path(self):
        if shutil.which("ffmpeg") is None:
            print("\n" + "=" * 80)
            print("WARNING: FFmpeg not found in system PATH.")
            print("Please ensure FFmpeg is installed and its 'bin' directory is added to your system's PATH.")
            print("=" * 80 + "\n")

    def _get_resolution_info(self, width, height):
        sorted_map = sorted(self.RESOLUTION_MAP, key=lambda x: x[0] * x[1], reverse=True)
        for res_w, res_h, label, min_score, max_score in sorted_map:
            if res_w == 0 and res_h == 0:
                continue
            if (width >= res_w and height >= res_h) or (height >= res_w and width >= res_h):
                return label, min_score, max_score
        return sorted_map[-1][2], sorted_map[-1][3], sorted_map[-1][4]

    def extract_citnow_metadata(self, url):
        print("🌐 Extracting CitNow page metadata...")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator="\n")

            metadata = {
                "page_url": url,
                "extraction_timestamp": datetime.now().isoformat(),
                "brand": None,
                "dealership": None,
                "vehicle": None,
                "registration": None,
                "vin": None,
                "service_advisor": None,
                "email": None,
                "phone": None,
            }

            # Brand detection by title/img
            title = soup.title.string if soup.title else ""
            if "BMW" in title or soup.find("img", src=lambda x: x and "bmw" in x.lower()):
                metadata["brand"] = "BMW"
            elif "MINI" in title or soup.find("img", src=lambda x: x and "mini" in x.lower()):
                metadata["brand"] = "MINI"
            elif "MG" in title or soup.find("img", src=lambda x: x and "mg" in x.lower()):
                metadata["brand"] = "MG"

            # Dealership extraction
            metadata["dealership"] = self._extract_dealership(soup, page_text)

            # TABLE-BASED extraction
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).replace(":", "").strip().lower()
                        value = cells[1].get_text(strip=True)

                        if any(kw in label for kw in ["vin", "chassis"]):
                            if re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", value):
                                metadata["vin"] = value
                        elif any(kw in label for kw in ["reg", "registration", "plate", "license"]):
                            metadata["registration"] = value
                        elif "vehicle" in label:
                            metadata["vehicle"] = value
                        elif any(kw in label for kw in ["advisor", "technician", "service", "presenter"]):
                            metadata["service_advisor"] = value
                        elif "email" in label:
                            email_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", value)
                            if email_match:
                                metadata["email"] = email_match.group(1)
                        elif "phone" in label or "mobile" in label:
                            phone_match = re.search(r"(\+?\d[\d\s-]{8,})", value)
                            if phone_match:
                                metadata["phone"] = phone_match.group(1)

            # Fallback: regex search in full plain text
            if not metadata["service_advisor"]:
                advisor_match = re.search(r"(?:Technician|Service Advisor)[\s:]*([A-Za-z]+(?:,?\s+[A-Za-z]+)*)", page_text, re.IGNORECASE)
                if advisor_match:
                    metadata["service_advisor"] = advisor_match.group(1).strip()

            if not metadata["email"]:
                email_match = re.search(r"Email[\s:]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", page_text, re.IGNORECASE)
                if email_match:
                    metadata["email"] = email_match.group(1).strip()

            if not metadata["phone"]:
                phone_match = re.search(r"Phone[\s:]*(\+?\d[\d\s-]{8,})", page_text, re.IGNORECASE)
                if phone_match:
                    metadata["phone"] = phone_match.group(1).strip()

            if not metadata["vin"]:
                vin_match = re.search(r"\b([A-HJ-NPR-Z0-9]{17})\b", page_text)
                if vin_match:
                    metadata["vin"] = vin_match.group(1).strip()
            if not metadata["registration"]:
                reg_match = re.search(r"(?:Registration|Reg\.?|Plate)\s*[:\-]*\s*([A-Za-z0-9\-\s]{5,15})", page_text)
                if reg_match:
                    maybe_reg = reg_match.group(1).strip()
                    if len(maybe_reg) <= 15:
                        metadata["registration"] = maybe_reg

            video_url = self._extract_video_url_from_page(soup, page_text)
            if video_url:
                metadata["video_url"] = video_url

            print("✅ Metadata extracted successfully!")
            return metadata
        except Exception as e:
            print(f"⚠️ Error extracting metadata: {e}")
            return {
                "page_url": url,
                "error": str(e),
                "extraction_timestamp": datetime.now().isoformat(),
            }

    def _extract_dealership(self, soup, page_text):
        candidates = set()
        keywords = ["Private Limited", "Pvt Ltd", "Motors", "Cars", "Dealer", "Automotive", "Showroom", "LLP"]
        
        for tag in ["h1", "h2", "h3"]:
            for el in soup.find_all(tag):
                t = el.get_text(separator=" ", strip=True)
                if any(kw in t for kw in keywords) and 8 < len(t) < 80:
                    candidates.add(t.split("\n")[0].strip())
        
        m = re.search(r"\bfrom\s+([A-Z][A-Za-z0-9&.,\- ]{3,}(?:Pvt\.?\s?Ltd|Private Limited|Motors|Cars|Automotive|Showroom|LLP|Limited|Services|Ahmedabad|Chennai|Bangalore|Delhi|Mumbai)?)", page_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if len(val) < 80:
                candidates.add(val)
        
        for line in page_text.split("\n"):
            s = line.strip()
            if (any(kw in s for kw in keywords) and 6 < len(s) < 80 and not any(w in s for w in ["browser", "JavaScript", "support", "disable", "presentation"])):
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
            video_elem = soup.find("video", {"src": True})
            if video_elem:
                return self._clean_url(video_elem["src"])
            video_elem = soup.find("video")
            if video_elem:
                source = video_elem.find("source", {"src": True})
                if source:
                    return self._clean_url(source["src"])
            iframe = soup.find("iframe", {"src": True})
            if iframe and "video" in iframe["src"]:
                return self._clean_url(iframe["src"])
            
            mp4_pattern = r'(https?://[^\s"]+\.mp4)'
            mp4_matches = re.findall(mp4_pattern, page_text)
            if mp4_matches:
                return self._clean_url(mp4_matches[0])
            
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    video_patterns = [
                        r'videoUrl["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                        r'src["\']?\s*[:=]\s*["\']([^"\']+\.mp4)["\']',
                        r'file["\']?\s*[:=]\s*["\']([^"\']+\.mp4)["\']',
                        r'"url"\s*:\s*"([^"]+\.mp4)"',
                        r'"video"\s*:\s*"([^"]+\.mp4)"',
                    ]
                    for pattern in video_patterns:
                        match = re.search(pattern, script.string)
                        if match:
                            video_url = match.group(1)
                            video_url = self._clean_url(video_url)
                            if not video_url.startswith("http"):
                                video_url = "https://southasia.citnow.com/" + video_url.lstrip("/")
                            return video_url
            return None
        except Exception as e:
            print(f"⚠️ Error extracting video URL: {e}")
            return None

    def _clean_url(self, url):
        if not url:
            return None
        url = url.replace("\\/\\/", "//").replace("\\/", "/").replace("\\", "")
        if url.startswith("//"):
            url = "https:" + url
        elif not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    def download_citnow_video(self, url):
        print("📥 Attempting to download CitNow video...")
        metadata = self.extract_citnow_metadata(url)
        video_url = metadata.get("video_url")
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
                "format": "best[ext=mp4]/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
                "user_agent": "Mozilla/5.0",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("⬇️ Trying yt-dlp download...")
                info = ydl.extract_info(url, download=False)
                ext = info.get("ext", "mp4")
                video_path = os.path.join(temp_dir, f"{unique_filename}.{ext}")
                ydl.download([url])
                if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                    print("✅ Video downloaded successfully with yt-dlp!")
                    return video_path
                else:
                    raise Exception("yt-dlp downloaded file not found or invalid size.")
        except Exception as e:
            print(f"⚠️ yt-dlp download failed: {e}")
        
        video_id_match = re.search(r"/([a-zA-Z0-9]+)$", url)
        if video_id_match:
            video_id = video_id_match.group(1)
            possible_urls = [
                f"https://southasia.citnow.com/videos/{video_id}.mp4",
                f"https://southasia.citnow.com/video/{video_id}/video.mp4",
                f"https://cdn.citnow.com/{video_id}.mp4",
                f"https://lts.in.prod.citnow.com/cin-southasia/{video_id}/output-1200k.mp4",
            ]
            for test_url in possible_urls:
                try:
                    print(f"🔍 Trying fallback URL: {test_url}")
                    return self._download_from_url(test_url)
                except Exception:
                    continue
        raise Exception("Could not download video from CitNow using any method.")

    def load_summarization_model(self):
        """Load summarization model only when needed"""
        if self.summarization_model is None:
            print("📝 Loading BART summarization model...")
            self.summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            self.summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    def load_translation_model(self, target_lang: str):
        """
        Loads the translation model for the specified target_lang from cache,
        or initializes it if not already loaded. Returns (model, tokenizer).
        """
        if target_lang == "en":
            print("🌍 Target language is English - no translation model needed.")
            return None, None # No model needed for English pass-through

        # Check if model for this language is already in cache
        if target_lang in self._translation_models:
            print(f"✅ Translation model for {target_lang} already loaded (from cache).")
            return self._translation_models[target_lang]

        print(f"🌍 Loading new translation model for {target_lang}...")
        model_name = self.lang_model_map.get(target_lang, NLLB_MODEL)

        try:
            if model_name == NLLB_MODEL:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else: # MarianMT fallback if other models are added
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

            self._translation_models[target_lang] = (model, tokenizer)
            print(f"✅ Translation model for {target_lang} loaded and cached.")
            return model, tokenizer
        except Exception as e:
            print(f"❌ Error loading translation model for {target_lang}: {e}")
            raise # Re-raise to indicate failure

    def extract_audio_from_video(self, video_path):
        """Extracts clean 16kHz mono WAV audio from a video file - Optimized for backend"""
        try:
            # Use temporary file for audio extraction
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            output_audio_path = temp_audio.name
            temp_audio.close()

            print(f"🔊 Extracting 16kHz mono audio to temporary file...")

            # ffmpeg command: convert to 16kHz mono PCM WAV
            command = [
                "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", 
                "-vn", "-y", output_audio_path
            ]

            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("✅ Audio extracted successfully.")
            return output_audio_path

        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed to extract audio:\n{e.stderr}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error during audio extraction: {e}")
            return None
    def analyze_audio_quality(self, audio_path):
        """Comprehensive audio quality analysis with detailed metrics"""
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            if len(y) == 0:
                return {
                    "prediction": "No Audio", "confidence": 0.0, "score": 0.0,
                    "issues": ["No audio detected"], "clarity_level": "Very Poor",
                    "detailed_analysis": {
                        "volume_level": "Silent", "noise_level": "Unknown", 
                        "speech_clarity": "Unknown", "background_noise": "Unknown",
                        "audio_balance": "Unknown"
                    }
                }
            
            # Extract comprehensive audio features
            features = {}
            
            # RMS Energy (Loudness)
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            features['rms_max'] = np.max(rms)
            features['rms_min'] = np.min(rms)
            
            # Zero Crossing Rate (Noisiness)
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Noise analysis
            S = np.abs(librosa.stft(y))
            spectral_flatness = librosa.feature.spectral_flatness(S=S)
            features['spectral_flatness_mean'] = np.mean(spectral_flatness)
            
            # Harmonic-Percussive separation
            harmonic, percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
            
            # MFCC for speech quality
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Silence detection
            frame_length = 2048
            hop_length = 512
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            silence_threshold = np.percentile(rms_frames, 25)
            silent_frames = np.sum(rms_frames < silence_threshold)
            total_frames = len(rms_frames)
            features['silence_ratio'] = silent_frames / total_frames if total_frames > 0 else 0
            
            # Detect specific issues
            issues = []
            detailed_scores = {}
            
            # Volume Analysis
            if features['rms_mean'] < 0.005:
                issues.append("Very low volume - hard to hear")
                detailed_scores['volume'] = 10
            elif features['rms_mean'] < 0.02:
                issues.append("Low volume - requires concentration")
                detailed_scores['volume'] = 40
            elif features['rms_mean'] < 0.05:
                issues.append("Moderate volume - acceptable but not ideal")
                detailed_scores['volume'] = 70
            elif features['rms_mean'] > 0.3:
                issues.append("Potential clipping - too loud")
                detailed_scores['volume'] = 60
            else:
                detailed_scores['volume'] = 90
            
            # Background Noise Analysis
            if features['zcr_mean'] > 0.2:
                issues.append("Heavy background noise - very distracting")
                detailed_scores['noise'] = 20
            elif features['zcr_mean'] > 0.12:
                issues.append("Moderate background noise - noticeable")
                detailed_scores['noise'] = 50
            elif features['zcr_mean'] > 0.08:
                issues.append("Light background noise - minimal impact")
                detailed_scores['noise'] = 75
            else:
                detailed_scores['noise'] = 95
            
            # Speech Clarity Analysis
            if features['spectral_flatness_mean'] > 0.9:
                issues.append("Poor speech clarity - very muffled")
                detailed_scores['clarity'] = 20
            elif features['spectral_flatness_mean'] > 0.7:
                issues.append("Reduced speech clarity - somewhat muffled")
                detailed_scores['clarity'] = 50
            elif features['spectral_flatness_mean'] > 0.5:
                issues.append("Average speech clarity")
                detailed_scores['clarity'] = 70
            else:
                detailed_scores['clarity'] = 90
            
            # Silence/Background Analysis
            if features['silence_ratio'] > 0.5:
                issues.append("Excessive silence or gaps")
                detailed_scores['consistency'] = 40
            elif features['silence_ratio'] > 0.3:
                issues.append("Some silence gaps")
                detailed_scores['consistency'] = 65
            else:
                detailed_scores['consistency'] = 85
            
            # Harmonic analysis for voice quality
            if features['harmonic_ratio'] < 0.3:
                issues.append("Poor voice quality - noisy")
                detailed_scores['voice_quality'] = 40
            elif features['harmonic_ratio'] < 0.6:
                detailed_scores['voice_quality'] = 70
            else:
                detailed_scores['voice_quality'] = 90
            
            # Calculate overall score with weighted components
            final_score = (
                detailed_scores['volume'] * 0.25 +
                detailed_scores['noise'] * 0.30 +
                detailed_scores['clarity'] * 0.25 +
                detailed_scores['consistency'] * 0.10 +
                detailed_scores['voice_quality'] * 0.10
            )
            final_score = np.clip(final_score, 0, 100)
            
            # Determine clarity level
            if final_score >= 90:
                clarity_level = "Excellent"
                prediction = "Studio Quality"
            elif final_score >= 80:
                clarity_level = "Very Good" 
                prediction = "Very Clear"
            elif final_score >= 70:
                clarity_level = "Good"
                prediction = "Clear"
            elif final_score >= 60:
                clarity_level = "Fair"
                prediction = "Acceptable"
            elif final_score >= 50:
                clarity_level = "Poor"
                prediction = "Noisy"
            elif final_score >= 30:
                clarity_level = "Very Poor"
                prediction = "Very Noisy"
            else:
                clarity_level = "Unusable"
                prediction = "Extremely Noisy"
            
            # Enhanced detailed analysis
            detailed_analysis = {
                "volume_level": "Very Low" if features['rms_mean'] < 0.02 else "Low" if features['rms_mean'] < 0.05 else "Ideal" if features['rms_mean'] < 0.2 else "Too Loud",
                "noise_level": "Heavy" if features['zcr_mean'] > 0.15 else "Moderate" if features['zcr_mean'] > 0.1 else "Light" if features['zcr_mean'] > 0.05 else "Minimal",
                "speech_clarity": "Poor" if features['spectral_flatness_mean'] > 0.8 else "Average" if features['spectral_flatness_mean'] > 0.6 else "Good" if features['spectral_flatness_mean'] > 0.4 else "Excellent",
                "background_noise": f"{features['silence_ratio']*100:.1f}% silence",
                "audio_balance": "Balanced" if 0.4 < features['harmonic_ratio'] < 0.8 else "Noisy" if features['harmonic_ratio'] <= 0.4 else "Clean",
                "dynamic_range": f"{features['rms_max']/max(features['rms_min'], 0.001):.1f}x"
            }
            
            return {
                "prediction": prediction,
                "confidence": round(float(final_score / 100), 3),
                "score": round(float(final_score), 1),
                "issues": issues,
                "clarity_level": clarity_level,
                "detailed_analysis": detailed_analysis,
                "component_scores": detailed_scores
            }
            
        except Exception as e:
            return {
                "prediction": "Analysis Failed",
                "confidence": 0.0,
                "score": 0.0,
                "issues": [f"Audio analysis error: {e}"],
                "clarity_level": "Unknown",
                "detailed_analysis": {}
            }
    def _get_video_duration_ffmpeg(self, video_path):
        """Get video duration using ffprobe as fallback"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration if duration > 0 else 0
        except Exception as e:
            print(f"⚠️ Could not determine duration with ffprobe: {e}")
            return 0

    def analyze_video_quality(self, video_path):
        """Comprehensive video quality analysis with stability, noise, and detail assessment"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "quality_score": 0.0, "quality_label": "Error",
                    "issues": ["Could not open video file"], "shake_level": "Unknown",
                    "resolution_quality": "Unknown", "detailed_analysis": {},
                    "component_scores": {}
                }
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # FIX 1: Better duration calculation with fallback
            if fps <= 0 or frame_count <= 0:
                duration = self._get_video_duration_ffmpeg(video_path)
                if duration > 0:
                    fps = 25  # Assume standard fps if unknown
                    frame_count = int(duration * fps)
                else:
                    return {
                        "quality_score": 0.0, "quality_label": "Invalid Video",
                        "issues": ["Cannot determine video duration"], "shake_level": "Unknown",
                        "resolution_quality": "Unknown", "detailed_analysis": {},
                        "component_scores": {}
                    }
            else:
                duration = frame_count / fps
            
            print(f"📊 Video Info: {width}x{height}, {fps:.1f} fps, {frame_count} frames, {duration:.1f}s")
            
            # Get resolution info
            res_label, min_res_score, max_res_score = self._get_resolution_info(width, height)
            
            # FIX 2: Dynamic frame sampling based on video length
            if duration > 300:  # Longer than 5 minutes
                num_frames_to_sample = min(150, frame_count)
            elif duration > 120:  # Longer than 2 minutes  
                num_frames_to_sample = min(80, frame_count)
            else:
                num_frames_to_sample = min(50, frame_count)
                
            print(f"🎯 Sampling {num_frames_to_sample} frames from {frame_count} total frames")
            
            sample_idxs = np.linspace(0, frame_count - 1, num_frames_to_sample, dtype=int)
            # Enhanced analysis variables
            sharpness_vals, brightness_vals, contrast_vals = [], [], []
            color_saturation_vals = []
            blockiness_vals = []  # Compression artifacts
            issues = []
            component_scores = {}
            
            # Enhanced shake and noise detection
            shake_score, shake_details = self._calculate_detailed_shake(video_path, sample_idxs)
            noise_score, noise_details = self._calculate_detailed_noise(video_path, sample_idxs)
            
            # Frame-by-frame comprehensive analysis
            prev_frame_gray = None
            frozen_frames = 0
            grabbed_frames = 0
            motion_changes = []
            
            for idx in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                grabbed_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Sharpness (Laplacian variance)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_vals.append(sharpness)
                
                # Brightness
                brightness = np.mean(gray)
                brightness_vals.append(brightness)
                
                # Contrast
                contrast = np.std(gray)
                contrast_vals.append(contrast)
                
                # Color saturation (from BGR to HSV)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
                color_saturation_vals.append(saturation)
                
                # Blockiness detection (compression artifacts)
                blockiness = self._detect_blockiness(gray)
                blockiness_vals.append(blockiness)
                
                # Motion analysis
                if prev_frame_gray is not None:
                    diff = cv2.absdiff(gray, prev_frame_gray)
                    motion_change = np.mean(diff)
                    motion_changes.append(motion_change)
                    if motion_change < 2.0:  # Very little change between frames
                        frozen_frames += 1
                
                prev_frame_gray = gray
            
            cap.release()
            
            if grabbed_frames < 10:
                return {
                    "quality_score": 10.0, "quality_label": "Very Poor",
                    "issues": ["Too few valid frames for analysis"], "shake_level": "Unknown",
                    "resolution_quality": res_label, "detailed_analysis": {},
                    "component_scores": {}
                }
            
            # Enhanced outlier removal
            def remove_outliers_enhanced(data):
                if len(data) < 3:
                    return data
                data = np.array(data)
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                if iqr == 0:
                    return data
                mask = (data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)
                return data[mask] if np.sum(mask) >= 3 else data
            
            # Calculate enhanced normalized scores
            sharpness_clean = remove_outliers_enhanced(sharpness_vals)
            brightness_clean = remove_outliers_enhanced(brightness_vals)
            contrast_clean = remove_outliers_enhanced(contrast_vals)
            saturation_clean = remove_outliers_enhanced(color_saturation_vals)
            blockiness_clean = remove_outliers_enhanced(blockiness_vals)
            
            sharpness_avg = np.mean(sharpness_clean) if len(sharpness_clean) > 0 else 0
            brightness_avg = np.mean(brightness_clean) if len(brightness_clean) > 0 else 0
            contrast_avg = np.mean(contrast_clean) if len(contrast_clean) > 0 else 0
            saturation_avg = np.mean(saturation_clean) if len(saturation_clean) > 0 else 0
            blockiness_avg = np.mean(blockiness_clean) if len(blockiness_clean) > 0 else 0
            
            # Enhanced scoring with better normalization
            sharpness_score = np.clip((sharpness_avg - 10) / (200 - 10) * 100, 0, 100)
            brightness_score = np.clip(100 - abs(brightness_avg - 127) / 64 * 100, 0, 100)
            contrast_score = np.clip((contrast_avg - 15) / (80 - 15) * 100, 0, 100)
            color_score = np.clip((saturation_avg - 30) / (100 - 30) * 100, 0, 100)
            compression_score = np.clip(100 - (blockiness_avg * 200), 0, 100)  # Lower blockiness is better
            
            # Motion consistency score
            if motion_changes:
                motion_consistency = np.std(motion_changes) if len(motion_changes) > 1 else 0
                motion_score = np.clip(100 - (motion_consistency * 10), 0, 100)
            else:
                motion_score = 50
            
            # Frozen frames penalty
            frozen_ratio = frozen_frames / grabbed_frames
            if frozen_ratio > 0.4:
                frozen_penalty = 30
            elif frozen_ratio > 0.2:
                frozen_penalty = 15
            else:
                frozen_penalty = 0
            
            # Enhanced component scoring
            component_scores = {
                'sharpness': sharpness_score, 'brightness': brightness_score,
                'contrast': contrast_score, 'color': color_score,
                'stability': shake_score, 'cleanliness': noise_score,
                'compression': compression_score, 'motion': motion_score
            }
            
            # Enhanced issue detection
            issues = []

            if sharpness_score < 15:
                issues.append("Very blurry - details hard to see")
            elif sharpness_score < 35:
                issues.append("Slightly soft or out of focus")

            if brightness_score < 40:
                if brightness_avg < 80:
                    issues.append("Video is too dark")
                else:
                    issues.append("Video is overexposed or washed out")
            elif brightness_score < 60:
                issues.append("Suboptimal lighting")

            if contrast_score < 40:
                issues.append("Low contrast - flat image")

            if color_score < 20: # Was 40
                issues.append("Dull colors - poor saturation")

            # Make shake issues less sensitive
            if shake_score < 40: # Was 30
                issues.append("Severe camera shake - very unstable")
            elif shake_score < 60: # Was 50
                issues.append("Noticeable camera shake")

            if noise_score < 40:
                issues.append("Visible visual noise/grain")

            if compression_score < 50:
                issues.append("Visible compression artifacts")

            if frozen_ratio > 0.3:
                issues.append("Many frozen/static frames")
            elif frozen_ratio > 0.15:
                issues.append("Some frozen frames")

            
            # Calculate overall score with enhanced weights
            resolution_base = (min_res_score + max_res_score) / 2
            
            visual_score = (
                sharpness_score * 0.30 +  # Most important
                brightness_score * 0.20 + # Very important
                contrast_score * 0.15 +
                noise_score * 0.15 +      # Cleanliness is important
                shake_score * 0.05 +      # Drastically reduced weight
                color_score * 0.05 +      # Less critical for this context
                compression_score * 0.05 +
                motion_score * 0.05
            )

            
            final_score = resolution_base * 0.25 + visual_score * 0.75
            final_score = np.clip(final_score - frozen_penalty, 0, 100)
            
            # Enhanced shake level classification
            if shake_score >= 85:
                shake_level = "Very Stable"
            elif shake_score >= 70:
                shake_level = "Stable"
            elif shake_score >= 55:
                shake_level = "Slightly Shaky"
            elif shake_score >= 40:
                shake_level = "Shaky"
            elif shake_score >= 25:
                shake_level = "Very Shaky"
            else:
                shake_level = "Extremely Shaky"
            
            # Enhanced detailed analysis
            detailed_analysis = {
                "resolution": f"{width}x{height} ({res_label})", "duration": f"{duration:.1f}s",
                "frame_rate": f"{fps:.1f} fps", "sharpness": f"{sharpness_score:.1f}%",
                "brightness": f"{brightness_score:.1f}%", "contrast": f"{contrast_score:.1f}%",
                "color_vibrancy": f"{color_score:.1f}%", "stability": f"{shake_score:.1f}%",
                "cleanliness": f"{noise_score:.1f}%", "compression": f"{compression_score:.1f}%",
                "motion_consistency": f"{motion_score:.1f}%", "frozen_frames": f"{frozen_ratio*100:.1f}%"
            }
            
            return {
                "quality_score": round(float(final_score), 1),
                "quality_label": self._get_quality_label(final_score),
                "issues": issues, "shake_level": shake_level,
                "resolution_quality": res_label, "detailed_analysis": detailed_analysis,
                "component_scores": component_scores
            }
            
        except Exception as e:
            return {
                "quality_score": 0.0, "quality_label": "Analysis Failed",
                "issues": [f"Video analysis error: {e}"], "shake_level": "Unknown",
                "resolution_quality": "Unknown", "detailed_analysis": {},
                "component_scores": {}
            }

    def _detect_blockiness(self, gray_frame):
        """Detect compression blockiness artifacts"""
        try:
            # Simple blockiness detection using variance of 8x8 blocks
            h, w = gray_frame.shape
            block_size = 8
            block_variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray_frame[i:i+block_size, j:j+block_size]
                    block_variances.append(np.var(block))
            
            if block_variances:
                # Low variance between blocks indicates compression artifacts
                return np.std(block_variances) / (np.mean(block_variances) + 1e-6)
            return 0
        except:
            return 0

    def _calculate_detailed_shake(self, video_path, sample_idxs):
        """Enhanced shake detection with detailed metrics"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 50.0, "Cannot analyze shake"
            
            lk_params = dict(
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            
            total_displacement = 0
            frame_pairs = 0
            max_displacement = 0
            displacements = []
            prev_gray = None
            
            for idx in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                    if p0 is not None:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]
                            if len(good_new) > 5:
                                displacement = np.mean(np.linalg.norm(good_new - good_old, axis=1))
                                total_displacement += displacement
                                displacements.append(displacement)
                                max_displacement = max(max_displacement, displacement)
                                frame_pairs += 1
                
                prev_gray = gray
            
            cap.release()
            
            if frame_pairs == 0:
                return 50.0, "Insufficient frames for shake analysis"
            
            avg_shake_px = total_displacement / frame_pairs
            shake_variance = np.std(displacements) if len(displacements) > 1 else 0
            
            # More sophisticated shake scoring
            base_score = 100 - (avg_shake_px / self.SHAKE_TOLERANCE_PX) * 50
            variance_penalty = min(shake_variance * 10, 20)  # Penalize inconsistent shake
            max_shake_penalty = min(max_displacement * 5, 15)  # Penalize worst-case shake
            
            shake_score = max(0, base_score - variance_penalty - max_shake_penalty)
            shake_score = np.clip(shake_score, 0, 100)
            
            shake_detail = f"Average movement: {avg_shake_px:.2f}px, Max: {max_displacement:.2f}px"
            
            return shake_score, shake_detail
            
        except Exception as e:
            return 50.0, f"Shake analysis error: {e}"

    def _calculate_detailed_noise(self, video_path, sample_idxs):
        """Enhanced noise detection with detailed metrics"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 50.0, "Cannot analyze noise"
            
            noise_estimates = []
            brightness_variations = []
            
            for idx in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Laplacian-based noise estimation
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                if laplacian.size > 0:
                    mad = np.median(np.abs(laplacian - np.median(laplacian)))
                    if mad > 0:
                        noise_estimates.append(mad)
                
                # Brightness variation (can indicate compression artifacts)
                brightness_variations.append(np.std(gray))
            
            cap.release()
            
            if not noise_estimates:
                return 50.0, "No noise estimates available"
            
            avg_noise_mad = np.mean(noise_estimates)
            brightness_variation = np.mean(brightness_variations) if brightness_variations else 0
            
            # Combined noise scoring
            base_score = 100 - (avg_noise_mad / self.NOISE_THRESHOLD_STD) * 50
            variation_penalty = min(brightness_variation * 2, 15)  # Penalize compression artifacts
            
            noise_score = max(0, base_score - variation_penalty)
            noise_score = np.clip(noise_score, 0, 100)
            
            noise_detail = f"Noise level: {avg_noise_mad:.2f}, Variation: {brightness_variation:.2f}"
            
            return noise_score, noise_detail
            
        except Exception as e:
            return 50.0, f"Noise analysis error: {e}"
        
    def calculate_overall_quality(self, audio_analysis, video_analysis):
        """Calculate overall quality out of 10 with equal importance for audio and video"""
        try:
            # Get individual scores (assuming they are out of 10 or 100)
            audio_score = audio_analysis.get('score', 0)
            video_score = video_analysis.get('quality_score', 0)
            
            # Normalize if needed (if scores are out of 100, convert to 10 scale)
            if audio_score > 10:
                audio_score = audio_score / 10.0
            if video_score > 10:
                video_score = video_score / 10.0
            
            # Equal weighting (50/50)
            audio_weight = 0.5
            video_weight = 0.5
            
            overall_score = (audio_score * audio_weight) + (video_score * video_weight)
            
            # Quality labels for 0–10 range
            if overall_score >= 9:
                overall_label = "Excellent"
            elif overall_score >= 8:
                overall_label = "Very Good"
            elif overall_score >= 7:
                overall_label = "Good"
            elif overall_score >= 6:
                overall_label = "Fair"
            elif overall_score >= 5:
                overall_label = "Poor"
            elif overall_score >= 3:
                overall_label = "Very Poor"
            else:
                overall_label = "Unusable"
            
            return {
                "overall_score": round(overall_score, 1),
                "overall_label": overall_label,
                "audio_contribution": round(audio_score * audio_weight, 1),
                "video_contribution": round(video_score * video_weight, 1),
                "breakdown": {
                    "audio_quality": round(audio_score, 1),
                    "video_quality": round(video_score, 1)
                }
            }
        except Exception as e:
            return {
                "overall_score": 0,
                "overall_label": "Calculation Failed",
                "error": str(e)
            }
    def load_pretrained_models(self):
        """Load all models at startup for better performance"""
        print("🔄 Loading all pretrained models at startup...")
        
        try:
            # Load summarization model
            self.load_summarization_model()
            print("✅ Summarization model loaded")
            
            # Pre-load common translation models into the cache
            # For NLLB, English is usually the source, so we only need target language models
            # Consider which languages are most likely to be requested first
            common_target_languages = ["hi", "te", "bn"] # Example: Hindi, Telugu, Bengali
            for lang_code in common_target_languages:
                try:
                    # Use the new load_translation_model signature
                    self.load_translation_model(lang_code)
                    print(f"✅ Translation model for {lang_code} pre-loaded.")
                except Exception as e:
                    print(f"⚠️ Could not pre-load translation model for {lang_code}: {e}")
            
            # Load faster-whisper model
            try:
                from faster_whisper import WhisperModel
                print("🎙️ Loading faster-whisper model...")
                # Consider using a smaller model for faster startup if 'large-v3' is too slow
                # Or make device/compute_type configurable
                self.faster_whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8") 
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"⚠️ Could not pre-load whisper model: {e}")
                
            print("🎯 All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading some models: {e}")
            # Don't raise the exception, continue without some models

    def transcribe_audio(self, audio_path, transcription_language=None, task='transcribe'):
        """High-performance transcription with minimal anti-repetition"""
        try:
            from faster_whisper import WhisperModel
            
            print(f"\n🔄 Starting faster-whisper transcription...")
            print(f"🌐 Language: {transcription_language or 'auto-detection'}")

            # Load faster-whisper model (if not already loaded)
            if self.faster_whisper_model is None:
                print("🎙️ Loading faster-whisper large-v3...")
                self.faster_whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            
            # Handle language parameter for Whisper
            # If transcription_language is 'auto' or None, Whisper will detect it.
            # Otherwise, force it to the specified language.
            whisper_lang_param = transcription_language if transcription_language != "auto" else None

            # Use reliable parameters from your previous working version
            segments, info = self.faster_whisper_model.transcribe(
                audio_path,
                language=whisper_lang_param, # Pass the resolved language parameter
                beam_size=5,
                best_of=5,
                temperature=0.0,  # More deterministic
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=300
                ),
                no_speech_threshold=0.8,
                repetition_penalty=1.0,
            )

            print(f"🔊 Detected language: '{info.language}' (probability: {info.language_probability:.2f})")

            # Simple anti-repetition - only remove exact duplicates
            transcription_parts = []
            last_segment = ""
            
            for segment in segments:
                clean_text = segment.text.strip()
                
                if not clean_text:
                    continue
                    
                # Only skip if it's an exact duplicate of the previous segment
                if clean_text != last_segment:
                    transcription_parts.append(clean_text)
                    last_segment = clean_text
                    # print(f"📝 Segment: {clean_text}") # Suppress verbose segment printing
                else:
                    print(f"🔄 Skipping exact duplicate: {clean_text}")

            if not transcription_parts:
                return "No clear speech detected in audio"

            full_transcription = " ".join(transcription_parts).strip()
            
            print(f"✅ Transcription successful! Length: {len(full_transcription)} characters")
            # print(f"📝 Final transcription: {full_transcription}") # Suppress full transcription printing
            
            return full_transcription

        except Exception as e:
            error_msg = f"faster-whisper transcription error: {e}"
            print(f"❌ {error_msg}")
            return f"Transcription failed: {str(e)}"

    def _is_similar_text(self, text1, text2, threshold=0.9):  # Higher default threshold
        """Check if two texts are similar above a threshold - LESS AGGRESSIVE"""
        if not text1 or not text2:
            return False
        
        text1_clean = text1.lower().strip()
        text2_clean = text2.lower().strip()
        
        # Exact match
        if text1_clean == text2_clean:
            return True
        
        # Only consider substantial segments for substring checks
        if len(text1_clean) < 20 or len(text2_clean) < 20:
            return False
        
        # Check if one is substantial substring of another
        if text1_clean in text2_clean or text2_clean in text1_clean:
            overlap_ratio = min(len(text1_clean), len(text2_clean)) / max(len(text1_clean), len(text2_clean))
            return overlap_ratio > 0.7  # Require substantial overlap
        
        # Calculate similarity ratio with higher threshold
        import difflib
        similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()
        return similarity > threshold

    def _remove_final_repetitions(self, text):
        """Remove only obvious repetitive patterns at the end - LESS AGGRESSIVE"""
        if not text:
            return text
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 3:
            return text
        
        # Only check the last 2 sentences against previous ones
        clean_sentences = sentences[:]  # Start with all sentences
        
        # Check if last 1-2 sentences are clear repetitions of earlier ones
        final_check = []
        for i in range(min(2, len(clean_sentences))):
            current_idx = -(i+1)
            current_sentence = clean_sentences[current_idx]
            
            is_clear_repetition = False
            # Check against sentences before the last 3
            for prev_sentence in clean_sentences[:max(0, current_idx-3)]:
                if self._is_similar_text(current_sentence, prev_sentence, threshold=0.95):
                    is_clear_repetition = True
                    print(f"🗑️ Removing clear repetition: {current_sentence}")
                    break
                    
            if not is_clear_repetition:
                final_check.append(current_sentence)
        
        # Reconstruct with non-repetitive endings
        if final_check:
            return '. '.join(clean_sentences[:len(clean_sentences)-len(final_check)] + final_check[::-1]) + '.'
        else:
            return '. '.join(clean_sentences) + '.'
    def summarize_text(self, text, summary_type="medium"):
            """
            Generates a summary of the provided text, using hierarchical summarization
            for long inputs and configurable output length.
            
            Args:
                text (str): The input text to summarize.
                summary_type (str): Desired summary length/detail. Options: "short", "medium", "long".
                                    Defaults to "medium".
            Returns:
                str: The generated summary or an error message.
            """
            self.load_summarization_model()

            if len(text.strip()) < 50:
                return "Text too short for meaningful summary"

            # Define length presets based on summary_type
            length_presets = {
                "short": {"min_length": 30, "max_length": 80, "num_beams": 2},
                "medium": {"min_length": 50, "max_length": 180, "num_beams": 3}, # Increased num_beams slightly for better quality
                "long": {"min_length": 80, "max_length": 250, "num_beams": 4},   # More beams for potentially higher quality long summary
            }
            params = length_presets.get(summary_type, length_presets["medium"])
            
            # BART-large-cnn has an effective max input length of 1024 tokens
            MAX_MODEL_INPUT_TOKENS = self.summarization_tokenizer.model_max_length # Get it directly from tokenizer
            if MAX_MODEL_INPUT_TOKENS > 1024: # Cap it if model_max_length is reported higher than typical BART usage
                MAX_MODEL_INPUT_TOKENS = 1024 
                
            OVERLAP_TOKENS = 50 # Tokens to overlap between chunks for context

            try:
                # Tokenize the full text to determine its actual token length
                # `truncation=False` ensures we get all tokens
                full_token_ids = self.summarization_tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
                
                # If the text is longer than what the model can handle in one go
                if len(full_token_ids) > MAX_MODEL_INPUT_TOKENS:
                    print(f"⚠️ Input text is too long ({len(full_token_ids)} tokens). Applying hierarchical summarization for '{summary_type}' summary.")
                    
                    chunk_summaries = []
                    current_start_idx = 0
                    
                    while current_start_idx < len(full_token_ids):
                        chunk_end_idx = min(current_start_idx + MAX_MODEL_INPUT_TOKENS, len(full_token_ids))
                        current_chunk_tokens = full_token_ids[current_start_idx:chunk_end_idx]
                        chunk_text = self.summarization_tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
                        
                        # Summarize each chunk
                        chunk_inputs = self.summarization_tokenizer([chunk_text], max_length=MAX_MODEL_INPUT_TOKENS, return_tensors="pt", truncation=True)
                        
                        # Generate a shorter summary for each chunk to avoid combining too much text
                        chunk_summary_ids = self.summarization_model.generate(
                            chunk_inputs["input_ids"],
                            max_length=max(params["min_length"] // 2, 20), # Aim for shorter chunk summaries, min 20
                            min_length=min(params["min_length"] // 4, 10),  # Min 10
                            length_penalty=1.5,
                            num_beams=params["num_beams"],
                            early_stopping=True,
                        )
                        chunk_summaries.append(self.summarization_tokenizer.decode(chunk_summary_ids[0], skip_special_tokens=True))
                        
                        # Move the window, ensuring overlap
                        current_start_idx += MAX_MODEL_INPUT_TOKENS - OVERLAP_TOKENS
                        if current_start_idx >= len(full_token_ids): # Handle case where last chunk might be too small
                            break

                    # Combine the mini-summaries
                    combined_summaries_text = " ".join(chunk_summaries)
                    print(f"✨ Summarizing {len(chunk_summaries)} chunk summaries into final summary...")
                    
                    # If the combined summaries are still too long, recursively summarize or truncate
                    final_inputs = self.summarization_tokenizer(
                        [combined_summaries_text], 
                        max_length=MAX_MODEL_INPUT_TOKENS, # Max input length for the final summarization step
                        return_tensors="pt", 
                        truncation=True
                    )
                    
                    summary_ids = self.summarization_model.generate(
                        final_inputs["input_ids"],
                        max_length=params["max_length"],
                        min_length=params["min_length"],
                        length_penalty=1.5,
                        num_beams=params["num_beams"],
                        early_stopping=True,
                    )
                    summary = self.summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    return summary

                else: # Text fits within the model's max input length, proceed with single pass
                    print(f"✅ Input text fits model's max length. Generating '{summary_type}' summary.")
                    inputs = self.summarization_tokenizer(
                        [text], max_length=MAX_MODEL_INPUT_TOKENS, return_tensors="pt", truncation=True
                    )

                    summary_ids = self.summarization_model.generate(
                        inputs["input_ids"],
                        max_length=params["max_length"],
                        min_length=params["min_length"],
                        length_penalty=1.5,
                        num_beams=params["num_beams"],
                        early_stopping=True,
                    )

                    summary = self.summarization_tokenizer.decode(
                        summary_ids[0], skip_special_tokens=True
                    )
                    return summary

            except Exception as e:
                print(f"❌ Summarization error: {e}")
                return f"Summarization error: {e}"


    def translate_text(self, text, target_language=None):
        # 1. Determine the effective target language for this specific translation call.
        actual_target_lang = target_language if target_language is not None else self.target_language
        
        print(f"🌍 Initiating translation to {actual_target_lang.upper()}...")

        # 2. If target is English, return original text (no translation needed)
        if actual_target_lang == "en":
            print(f"🎯 Target language is English - returning original text without translation.")
            return text
        
        # 3. Load the specific model/tokenizer for this target language from cache (or initialize)
        try:
            translation_model, translation_tokenizer = self.load_translation_model(actual_target_lang)
            if translation_model is None: # Should only happen if target_lang was 'en' but somehow got here
                return text
        except Exception as e:
            return f"Translation failed due to model loading error for {LANGUAGE_NAME_LOOKUP.get(actual_target_lang, actual_target_lang)}: {e}"

        # 4. Input text validation and truncation
        if len(text.strip()) < 10:
            return "Input text too short to translate"
        if len(text) > 1500:
            text = text[:1500] + "..."

        try:
            # All our Indian languages use NLLB, so this branch will always be taken
            # when actual_target_lang is not 'en'
            model_name = self.lang_model_map.get(actual_target_lang, NLLB_MODEL)

            if model_name == NLLB_MODEL:
                # CRITICAL: Tell the tokenizer the source language is English
                translation_tokenizer.src_lang = "eng_Latn"

                # Get the FLORES code for the target language
                flores_code = INDIAN_LANGUAGE_CODES.get(actual_target_lang, "eng_Latn")
                forced_bos_token_id = translation_tokenizer.convert_tokens_to_ids(flores_code)
                if not isinstance(forced_bos_token_id, int):
                    forced_bos_token_id = forced_bos_token_id[0] if forced_bos_token_id else None

                if forced_bos_token_id is None:
                    print(f"⚠️ Warning: Could not find valid BOS token for {actual_target_lang}. Defaulting to English token for translation output.")
                    forced_bos_token_id = translation_tokenizer.convert_tokens_to_ids("eng_Latn")
                    if isinstance(forced_bos_token_id, list): forced_bos_token_id = forced_bos_token_id[0]

                inputs = translation_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=400,
                )
                generated_tokens = translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=400,
                    num_beams=2,
                    early_stopping=True,
                )
                translated_text = translation_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
                
                print(f"✅ Translation completed for {LANGUAGE_NAME_LOOKUP.get(actual_target_lang, actual_target_lang)}")
                return translated_text
                
            else: # Fallback for other MarianMT models if you had them
                inputs = translation_tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=400,
                )
                translated = translation_model.generate(
                    **inputs,
                    max_length=400,
                    num_beams=2,
                    early_stopping=True,
                )
                translated_text = translation_tokenizer.decode(
                    translated[0], skip_special_tokens=True
                )
                
                print(f"✅ MarianMT translation completed for {LANGUAGE_NAME_LOOKUP.get(actual_target_lang, actual_target_lang)}")
                return translated_text

        except Exception as e:
            print(f"❌ Translation error for {actual_target_lang}: {e}")
            return f"Translation failed for {LANGUAGE_NAME_LOOKUP.get(actual_target_lang, actual_target_lang)}: {e}"


    def process_video(self, video_input, transcription_language=None, target_language_short=None):
        # The line below is now only for logging/reporting the initial request,
        # not for controlling the *active* translation model state within the Analyzer.
        # The target_language for translation is passed directly to translate_text().
        requested_target_language = target_language_short if target_language_short else self.target_language

        print(f"\n🌍 TARGET LANGUAGE FOR THIS JOB: {requested_target_language.upper()}")
        print("\n" + "=" * 60)
        print("🚀 ENHANCED UNIFIED VIDEO ANALYSIS PIPELINE")
        print("=" * 60)
        results = {
            "input_source": video_input,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_steps": [],
            "target_language": requested_target_language, # Use the correctly set requested_target_language for results
            "transcription_language": transcription_language or "auto",
        }
        temp_files_to_clean = []
        try:
            if isinstance(video_input, str) and "citnow.com" in video_input:
                print("\n📊 EXTRACTING CITNOW METADATA")
                print("-" * 40)
                metadata = self.extract_citnow_metadata(video_input)
                results["citnow_metadata"] = metadata
                results["processing_steps"].append("metadata_extraction")
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

            # This correctly extracts the full audio and gets its path.
            audio_path = self.extract_audio_from_video(video_path)

            if not audio_path:
                raise ValueError("Audio extraction failed. Cannot proceed.")

            results["processing_steps"].append("audio_extraction")

            print("\n🎥 ANALYZING VIDEO QUALITY")
            print("-" * 40)
            results["video_analysis"] = self.analyze_video_quality(video_path)
            results["processing_steps"].append("video_quality_analysis")
            print(f"Quality: {results['video_analysis']['quality_label']} ({results['video_analysis']['quality_score']:.1f}/100)")

            print("\n🔊 ANALYZING AUDIO QUALITY")
            print("-" * 40)
            # Uses the full, correct audio path
            results["audio_analysis"] = self.analyze_audio_quality(audio_path)
            results["processing_steps"].append("audio_quality_analysis")
            print(f"Clarity: {results['audio_analysis']['prediction']}")

            # Calculate overall quality score
            print("\n📊 CALCULATING OVERALL QUALITY")
            print("-" * 40)
            overall_quality = self.calculate_overall_quality(results["audio_analysis"], results["video_analysis"])
            results["overall_quality"] = overall_quality
            results["processing_steps"].append("overall_quality_assessment")
            print(f"Overall Quality: {overall_quality['overall_label']} ({overall_quality['overall_score']:.1f}/10)")

            print("\n💬 CONVERTING SPEECH TO TEXT")
            print("-" * 40)
            print(f"📝 Transcription language: {transcription_language or 'auto-detection'}")
            transcription = self.transcribe_audio(
                audio_path, 
                transcription_language=transcription_language,
                task='transcribe'
            )
            results["transcription"] = {
                "text": transcription,
                "length": len(transcription),
                "language": transcription_language or "auto-detected",
            }
            results["processing_steps"].append("speech_to_text")
            print(f"✅ Full transcription completed ({len(transcription)} characters)")

            print("\n📝 GENERATING SUMMARY")
            print("-" * 40)
            summary = self.summarize_text(transcription)
            results["summarization"] = {
                "summary": summary,
                "length": len(summary),
                "reduction_ratio": f"{((1 - len(summary)/len(transcription)) * 100):.1f}%"
                if len(transcription) > 0
                else "N/A",
            }
            results["processing_steps"].append("text_summarization")
            print(f"Summary generated ({results['summarization']['reduction_ratio']} reduction)")

            # TRANSLATION
            print(f"\n🌍 TRANSLATING TO {requested_target_language.upper()}")
            print("-" * 40)
            # Pass the requested_target_language directly to translate_text
            translation = self.translate_text(transcription, target_language=requested_target_language)
            results["translation"] = {
                "translated_text": translation,
                "target_language": requested_target_language,  # Use the actual target language
                "length": len(translation),
            }
            results["processing_steps"].append("translation")
            print(f"Translation completed ({len(translation)} characters)")

        except Exception as pipeline_e:
            print(f"\n❌ Pipeline stopped due to an error: {pipeline_e}")
            results["error_message"] = str(pipeline_e)
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
            if input_source.startswith(("http://", "https://")):
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
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "video")
                ext = info.get("ext", "mp4")
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
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            ext = self._get_file_extension(url, response.headers.get("content-type", ""))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            with open(temp_file.name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return temp_file.name
        except Exception as e:
            raise Exception(f"URL download failed: {e}")

    def _get_file_extension(self, url, content_type):
        parsed = urlparse(url)
        path_ext = os.path.splitext(parsed.path)[1]
        if path_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            return path_ext
        if "mp4" in content_type:
            return ".mp4"
        elif "avi" in content_type:
            return ".avi"
        elif "quicktime" in content_type:
            return ".mov"
        elif "webm" in content_type:
            return ".webm"
        return ".mp4"

    def _get_quality_label(self, score):
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 25:
            return "Poor"
        else:
            return "Very Poor"

    def generate_comprehensive_report(self, results):
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE VIDEO ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {results['processing_timestamp']}")
        report.append(f"Target Language: {results['target_language'].upper()}")
        report.append(f"Processing Steps: {', '.join(results['processing_steps'])}")
        report.append("")
        
        if "citnow_metadata" in results:
            report.append("CITNOW SERVICE INFORMATION")
            report.append("-" * 60)
            meta = results["citnow_metadata"]
            report.append(f"Dealership: {meta.get('dealership', 'N/A')}")
            report.append(f"Vehicle Registration: {meta.get('vehicle', 'N/A')}")
            report.append(f"Service Advisor: {meta.get('service_advisor', 'N/A')}")
            report.append(f"Contact Email: {meta.get('email', 'N/A')}")
            report.append(f"Contact Phone: {meta.get('phone', 'N/A')}")
            report.append(f"Source URL: {meta.get('page_url', 'N/A')}")
            report.append("")
        
        if "video_analysis" in results:
            report.append("VIDEO QUALITY ANALYSIS")
            report.append("-" * 60)
            video = results["video_analysis"]
            report.append(f"Quality Score: {video.get('quality_score', 0):.1f}/100")
            report.append(f"Quality Label: {video.get('quality_label', 'N/A')}")
            report.append(f"Resolution: {video.get('resolution_quality', 'N/A')}")
            report.append(f"Camera Stability: {video.get('shake_level', 'N/A')}")
            
            if "detailed_analysis" in video:
                report.append("\nDetailed Metrics:")
                for metric, value in video["detailed_analysis"].items():
                    report.append(f"  - {metric.title()}: {value}")
            
            if "issues" in video and video["issues"]:
                report.append(f"\nDetected Issues ({len(video['issues'])}):")
                for i, issue in enumerate(video["issues"], 1):
                    report.append(f"  {i}. {issue}")
            else:
                report.append("\nNo major video issues detected.")
            report.append("")
        
        if "audio_analysis" in results:
            report.append("AUDIO ANALYSIS")
            report.append("-" * 60)
            audio = results["audio_analysis"]
            report.append(f"Clarity Level: {audio.get('clarity_level', 'N/A')}")
            report.append(f"Prediction: {audio.get('prediction', 'N/A')}")
            report.append(f"Confidence: {audio.get('confidence', 0):.2%}")
            report.append(f"Score: {audio.get('score', 0):.1f}/100")
            
            if "detailed_analysis" in audio:
                report.append("\nDetailed Audio Analysis:")
                for metric, value in audio["detailed_analysis"].items():
                    report.append(f"  - {metric.replace('_', ' ').title()}: {value}")
            
            if "issues" in audio and audio["issues"]:
                report.append(f"\nDetected Audio Issues ({len(audio['issues'])}):")
                for i, issue in enumerate(audio["issues"], 1):
                    report.append(f"  {i}. {issue}")
            else:
                report.append("\nNo major audio issues detected.")
            report.append("")
        
        if "overall_quality" in results:
            report.append("OVERALL QUALITY ASSESSMENT")
            report.append("-" * 60)
            overall = results["overall_quality"]
            report.append(f"Overall Score: {overall.get('overall_score', 0):.1f}/100")
            report.append(f"Overall Label: {overall.get('overall_label', 'N/A')}")
            report.append(f"Audio Contribution: {overall.get('audio_contribution', 0):.1f}")
            report.append(f"Video Contribution: {overall.get('video_contribution', 0):.1f}")
            report.append("")

        if "transcription" in results:
            report.append("FULL TRANSCRIPTION")
            report.append("-" * 60)
            report.append(results["transcription"]["text"])
            report.append("")
            report.append(f"Length: {results['transcription']['length']} characters")
            report.append("")
        if "summarization" in results:
            report.append("SUMMARY")
            report.append("-" * 60)
            report.append(results["summarization"]["summary"])
            report.append("")
            report.append(f"Reduction: {results['summarization']['reduction_ratio']}")
            report.append("")
        if "translation" in results:
            report.append(f"TRANSLATION ({results['target_language'].upper()})")
            report.append("-" * 60)
            report.append(results["translation"]["translated_text"])
            report.append("")
            report.append(f"Length: {results['translation']['length']} characters")
            report.append("")
        if "error_message" in results:
            report.append("ERROR DETAILS")
            report.append("-" * 60)
            report.append(f"Pipeline encountered an error: {results['error_message']}")
            report.append("")
        return "\n".join(report)

def main():
    # Custom JSON encoder that handles NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.str_):
                return str(obj)
            else:
                return super().default(obj)

    # Keep the analyzer and its loaded models in memory
    analyzer = None

    while True:
        citnow_url = input("Enter CitNow video URL (or type 'exit' to quit): ").strip()

        if citnow_url.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        print("\nAvailable Indian languages for both transcription and translation:")
        for code, name in LANGUAGE_NAME_LOOKUP.items():
            print(f"  {code} - {name}")

        transcription_language = input("Enter SPOKEN language code (e.g. hi, ta, 'auto'): ").strip().lower()
        target_language_short = input("Enter TARGET language code for translation (e.g. hi, ta): ").strip().lower()

        # Model Loading Optimization
        if analyzer is None:
            print("\nFirst run: Initializing and loading models...")
            analyzer = UnifiedMediaAnalyzer(target_language=target_language_short)
            # Ensure the initial target language is loaded if it's not 'en'
            analyzer.load_pretrained_models()
        else:
            # When subsequent analyses happen, the analyzer instance is reused.
            # The translate_text method now handles loading/switching models correctly
            # based on the target_language_short passed to process_video.
            print("\nReady for next analysis...")

        # Process the Video
        results = analyzer.process_video(
            citnow_url,
            transcription_language=transcription_language,
            target_language_short=target_language_short, # Pass the requested target language
        )

        # Save a unique report for each video
        report = analyzer.generate_comprehensive_report(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize the URL to create a valid filename
        safe_url_part = re.sub(r"[^a-zA-Z0-9]", "_", citnow_url.split("/")[-1])

        report_filename_txt = f"analysis_{safe_url_part}_{timestamp}.txt"
        report_filename_json = f"analysis_{safe_url_part}_{timestamp}.json"

        with open(report_filename_txt, "w", encoding="utf-8") as f:
            f.write(report)
        with open(report_filename_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        print(f"\n✅ Analysis complete! Reports saved as '{report_filename_txt}' and '{report_filename_json}'.")
        print("-" * 80)

if __name__ == "__main__":
    main()
