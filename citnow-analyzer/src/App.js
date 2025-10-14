// App.js (NO CHANGES NEEDED FOR THIS FILE)
import React, { useState } from 'react';
import { Box, Dialog, DialogContent, IconButton, Snackbar, Alert, Typography, Button } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import Navbar from './components/Navbar';
import AnalysisForm from './components/AnalysisForm';
import ResultsTable from './components/ResultTable';
import BulkUpload from './components/BulkUpload';
import AnalysisDetails from './components/AnalysisDetails'; // Ensure this is correctly displaying individual results

function App() {
  const [activeTab, setActiveTab] = useState('new');
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const handleSingleAnalysisComplete = (data) => {
    if (data && data.results) {
        const fullAnalysis = {
            _id: data.result_id,
            ...data.results,
        };
        setSelectedAnalysis(fullAnalysis);
        setShowDetails(true);
        setSnackbar({ open: true, message: 'Single analysis completed successfully!', severity: 'success' });
    } else {
        console.error("API response missing 'results' key:", data);
        setSnackbar({ open: true, message: 'Analysis completed, but no results found in response.', severity: 'warning' });
    }
  };

  const handleSelectAnalysis = (analysis) => {
    setSelectedAnalysis(analysis);
    setShowDetails(true);
  };

  const handleCloseDetails = () => {
    setShowDetails(false);
    setSelectedAnalysis(null);
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'new':
        return <AnalysisForm onAnalyze={handleSingleAnalysisComplete} />;
      case 'bulk':
        return <BulkUpload onSelectAnalysis={handleSelectAnalysis} />; 
      case 'history':
        return <ResultsTable onSelect={handleSelectAnalysis} />; // Correctly passes onSelect
      default:
        return <AnalysisForm onAnalyze={handleSingleAnalysisComplete} />;
    }
  };

  return (
    <Box sx={{ flexGrow: 1, minHeight: '100vh', backgroundColor: 'grey.50' }}>
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <Box sx={{ p: { xs: 2, md: 3 } }}>
        {renderContent()}
      </Box>

      <Dialog
        open={showDetails}
        onClose={handleCloseDetails}
        fullWidth={true}
        maxWidth="lg"
      >
        <DialogContent sx={{ position: 'relative', p: { xs: 2, sm: 3 } }}>
          <IconButton
            onClick={handleCloseDetails}
            sx={{
              position: 'absolute',
              right: 12,
              top: 12,
              color: (theme) => theme.palette.grey[500],
            }}
          >
            <CloseIcon />
          </IconButton>

          {selectedAnalysis && (
            <AnalysisDetails
              analysis={selectedAnalysis}
              onClose={handleCloseDetails}
            />
          )}
        </DialogContent>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default App;
