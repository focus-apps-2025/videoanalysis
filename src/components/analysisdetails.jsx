import React from "react";
import { FaFilePdf, FaCar, FaUser, FaPhone, FaEnvelope, FaVideo, FaMicrophone, FaFileAlt, FaLanguage } from "react-icons/fa";
import { generatePDF } from "./pdfgenerator";

export default function AnalysisDetails({ result, onBack }) {
  if (!result) return null;

  const metadata = result.citnow_metadata || {};
  const videoAnalysis = result.video_analysis || {};
  const audioAnalysis = result.audio_analysis || {};

  return (
    <div id="result-to-pdf" className="max-w-6xl mx-auto">
      <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-rose-100">
        {/* Header */}
        <div className="bg-gradient-to-r from-rose-50 to-rose-100 px-8 py-6 border-b border-rose-200">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-gray-800">Analysis Report</h2>
              <p className="text-gray-600">Comprehensive service video evaluation</p>
            </div>
            <button 
              onClick={onBack} 
              className="flex items-center text-rose-600 hover:text-rose-800 font-medium"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
              </svg>
              Back to Dashboard
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-8">
          {/* CitNow Info Section */}
          <div className="mb-10">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <FaCar className="text-rose-500 mr-2" /> CitNow Service Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <InfoCard label="Dealership" value={metadata.dealership} icon={<FaUser className="text-rose-500" />} />
              <InfoCard label="Vehicle" value={metadata.vehicle || metadata.registration || 'N/A'} icon={<FaCar className="text-rose-500" />} />
              <InfoCard label="Service Advisor" value={metadata.service_advisor || 'N/A'} icon={<FaUser className="text-rose-500" />} />
              <InfoCard label="VIN" value={metadata.vin || 'N/A'} icon={<FaFileAlt className="text-rose-500" />} />
              <InfoCard label="Email" value={metadata.email || 'N/A'} icon={<FaEnvelope className="text-rose-500" />} />
              <InfoCard label="Phone" value={metadata.phone || 'N/A'} icon={<FaPhone className="text-rose-500" />} />
            </div>
          </div>

          {/* Quality Analysis */}
          <div className="mb-10">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Quality Assessment</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <QualityCard 
                title="Video Quality" 
                score={videoAnalysis.quality_score} 
                label={videoAnalysis.quality_label}
                icon={<FaVideo className="text-rose-500 text-xl" />}
              />
              <QualityCard 
                title="Audio Quality" 
                score={audioAnalysis.confidence ? Math.round(audioAnalysis.confidence * 100) : 0}
                label={audioAnalysis.prediction || 'N/A'}
                icon={<FaMicrophone className="text-rose-500 text-xl" />}
              />
            </div>
          </div>

          {/* Transcription & Translation */}
          <div className="space-y-8">
            <ContentSection 
              title="Transcription" 
              content={result.transcription?.text || 'N/A'}
              icon={<FaFileAlt className="text-rose-500" />}
            />
            
            <ContentSection 
              title="Summary" 
              content={result.summarization?.summary || 'N/A'}
              icon={<FaFileAlt className="text-rose-500" />}
            />
            
            <ContentSection 
              title={`Translation (${result.target_language?.toUpperCase()})`} 
              content={result.translation?.translated_text || 'N/A'}
              icon={<FaLanguage className="text-rose-500" />}
            />
          </div>

          {/* PDF Button */}
          <div className="mt-10 pt-6 border-t border-gray-100 flex justify-center">
            <button 
              onClick={() => generatePDF("result-to-pdf", `citnow_${result._id}.pdf`)} 
              className="flex items-center bg-gradient-to-r from-rose-500 to-rose-600 hover:from-rose-600 hover:to-rose-700 text-white font-semibold px-6 py-3 rounded-xl transition-all shadow-lg"
            >
              <FaFilePdf className="mr-2" /> Download PDF Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const InfoCard = ({ label, value, icon }) => (
  <div className="bg-rose-50 rounded-xl p-4 border border-rose-100">
    <div className="flex items-center mb-2">
      <div className="mr-3 p-2 bg-white rounded-lg shadow-sm">
        {icon}
      </div>
      <span className="text-sm font-medium text-gray-600">{label}</span>
    </div>
    <p className="text-gray-800 font-medium">{value}</p>
  </div>
);

const QualityCard = ({ title, score, label, icon }) => (
  <div className="bg-gradient-to-br from-rose-50 to-white rounded-2xl p-6 border border-rose-100">
    <div className="flex items-center mb-4">
      <div className="p-3 bg-rose-100 rounded-xl mr-4">
        {icon}
      </div>
      <h4 className="text-lg font-semibold text-gray-800">{title}</h4>
    </div>
    <div className="flex items-end">
      <span className="text-4xl font-bold text-rose-600">{score || 0}</span>
      <span className="text-gray-600 ml-2">/100</span>
    </div>
    <div className="mt-2">
      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
        label === 'Excellent' ? 'bg-green-100 text-green-800' :
        label === 'Good' ? 'bg-blue-100 text-blue-800' :
        label === 'Fair' ? 'bg-yellow-100 text-yellow-800' :
        'bg-red-100 text-red-800'
      }`}>
        {label || 'N/A'}
      </span>
    </div>
  </div>
);

const ContentSection = ({ title, content, icon }) => (
  <div className="bg-rose-50 rounded-2xl p-6 border border-rose-100">
    <div className="flex items-center mb-4">
      <div className="p-2 bg-white rounded-lg shadow-sm mr-3">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
    </div>
    <div className="bg-white rounded-xl p-4 min-h-[100px] border border-rose-100 whitespace-pre-wrap">
      {content}
    </div>
  </div>
);