import React from "react";

export default function Navbar({ activeTab, onTabChange, onBack }) {
  return (
    <nav className="bg-white shadow-sm border-b border-rose-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="bg-rose-500 w-8 h-8 rounded-lg"></div>
              <span className="ml-3 text-xl font-bold text-gray-900">CitNow Analyzer</span>
            </div>
          </div>
          
          <div className="flex items-center">
            <div className="flex space-x-4">
              <button
                onClick={() => onTabChange('new')}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'new' 
                    ? 'bg-rose-100 text-rose-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                New Analysis
              </button>
              <button
                onClick={() => onTabChange('history')}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  activeTab === 'history' 
                    ? 'bg-rose-100 text-rose-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                History
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}