import React, { useState } from "react";
import Navbar from "./components/navbar";
import AnalysisForm from "./components/analysisform";
import ResultsTable from "./components/historylist";
import AnalysisDetails from "./components/analysisdetails";

export default function App() {
  const [selected, setSelected] = useState(null);
  const [current, setCurrent] = useState(null);
  const [activeTab, setActiveTab] = useState('new'); // 'new' or 'history'

  return (
    <div className="bg-gray-50 min-h-screen">
      <Navbar 
        activeTab={activeTab} 
        onTabChange={setActiveTab} 
        onBack={() => {
          setActiveTab('new');
          setCurrent(null);
          setSelected(null);
        }} 
      />
      
      <div className="container mx-auto px-4 py-8">
        {activeTab === 'new' && !current && (
          <>
            <AnalysisForm onAnalyze={setCurrent} />
          </>
        )}
        
        {current && (
          <AnalysisDetails 
            result={current} 
            onBack={() => {
              setCurrent(null);
              setActiveTab('new');
            }} 
          />
        )}
        
        {activeTab === 'history' && !selected && (
          <ResultsTable onSelect={setSelected} />
        )}
        
        {selected && (
          <AnalysisDetails 
            result={selected} 
            onBack={() => {
              setSelected(null);
              setActiveTab('history');
            }} 
          />
        )}
      </div>
    </div>
  );
}