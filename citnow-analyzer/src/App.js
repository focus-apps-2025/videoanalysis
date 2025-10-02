import React, { useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { luxuryTheme } from './theme';
import Navbar from './components/Navbar';
import AnalysisForm from './components/AnalysisForm';
import ResultsTable from './components/ResultTable';
import AnalysisDetails from './components/AnalysisDetails';

export default function App() {
  const [selected, setSelected] = useState(null);
  const [current, setCurrent] = useState(null);
  const [activeTab, setActiveTab] = useState('new');

  return (
    <ThemeProvider theme={luxuryTheme}>
      <CssBaseline />
      <Box sx={{ backgroundColor: 'background.default', minHeight: '100vh' }}>
        <Navbar 
          activeTab={activeTab} 
          onTabChange={setActiveTab} 
          onBack={() => {
            setActiveTab('new');
            setCurrent(null);
            setSelected(null);
          }} 
        />
        
        <Box sx={{ py: 4, px: 2 }}>
          {activeTab === 'new' && !current && (
            <AnalysisForm onAnalyze={setCurrent} />
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
        </Box>
      </Box>
    </ThemeProvider>
  );
}