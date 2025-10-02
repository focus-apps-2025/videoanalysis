import React, { useEffect, useState } from "react";
import axios from "axios";
import { FaCar, FaUser, FaFileAlt, FaVideo, FaMicrophone } from "react-icons/fa";

export default function ResultsTable({ onSelect }) {
  const [list, setList] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/results")
      .then(res => setList(res.data))
      .catch(err => console.error(err));
  }, []);

  const getStatusColor = (quality) => {
    if (!quality) return 'bg-gray-100 text-gray-800';
    if (quality >= 85) return 'bg-green-100 text-green-800';
    if (quality >= 70) return 'bg-blue-100 text-blue-800';
    if (quality >= 50) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-rose-100">
        <div className="bg-gradient-to-r from-rose-50 to-rose-100 px-8 py-6 border-b border-rose-200">
          <h2 className="text-2xl font-bold text-gray-800">Previous Analyses</h2>
          <p className="text-gray-600 mt-1">Review your historical service video evaluations</p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-rose-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dealership</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Vehicle</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Advisor</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">VIN</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Video Q</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Audio Q</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {list.length === 0 ? (
                <tr>
                  <td colSpan="8" className="px-6 py-4 text-center text-gray-500">
                    No previous analyses found
                  </td>
                </tr>
              ) : (
                list.map(r => {
                  const meta = r.citnow_metadata || {};
                  const video = r.video_analysis || {};
                  const audio = r.audio_analysis || {};
                  
                  return (
                    <tr key={r._id} className="hover:bg-rose-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FaUser className="text-rose-500 mr-2" />
                          <div className="text-sm font-medium text-gray-900">{meta.dealership || 'N/A'}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FaCar className="text-rose-500 mr-2" />
                          <div className="text-sm text-gray-900">{meta.vehicle || meta.registration || 'N/A'}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FaUser className="text-rose-500 mr-2" />
                          <div className="text-sm text-gray-900">{meta.service_advisor || 'N/A'}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FaFileAlt className="text-rose-500 mr-2" />
                          <div className="text-sm text-gray-900">{meta.vin || 'N/A'}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FaVideo className="text-rose-500 mr-2" />
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(video.quality_score)}`}>
                            {video.quality_score || 0}/100
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FaMicrophone className="text-rose-500 mr-2" />
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            audio.prediction === 'Clear' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                          }`}>
                            {audio.prediction || 'N/A'}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(r.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button 
                          onClick={() => onSelect(r)} 
                          className="text-rose-600 hover:text-rose-800 font-medium flex items-center"
                        >
                          View Details
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}