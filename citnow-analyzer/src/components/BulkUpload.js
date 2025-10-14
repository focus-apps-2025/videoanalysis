// BulkUpload.js - Complete with Stop/Delete functionality
import React, { useState, useEffect, useRef } from 'react';
import {
  Box, Paper, Typography, Button, LinearProgress,
  Card, CardContent, Alert, Chip, Grid, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow,
  Dialog, DialogTitle, DialogContent, DialogActions,
  IconButton, Tooltip, Snackbar, TextField,
  MenuItem,
} from '@mui/material';
import { 
  Upload, PlayArrow, CheckCircle, Error, TableChart, 
  Close, History, Refresh, Stop, Delete, Download,Language, Translate
} from '@mui/icons-material';

// Persistent storage for batch IDs
const STORAGE_KEY = 'bulkProcessingBatches';
const API_BASE_URL = 'http://localhost:5000';

// Language options (same as your AnalysisForm)
const LANGS = [
  { code: "auto", name: "Auto Detect" },
  { code:"en",  name:"English"},
  { code: 'as', name: 'Assamese' },
  { code: 'bn', name: 'Bengali' },
  { code: 'gu', name: 'Gujarati' },
  { code: 'hi', name: 'Hindi' },
  { code: 'kn', name: 'Kannada' },
  { code: 'ml', name: 'Malayalam' },
  { code: 'mr', name: 'Marathi' },
  { code: 'ne', name: 'Nepali' },
  { code: 'or', name: 'Odia' },
  { code: 'pa', name: 'Punjabi' },
  { code: 'sa', name: 'Sanskrit' },
  { code: 'ta', name: 'Tamil' },
  { code: 'te', name: 'Telugu' },
  { code: 'ur', name: 'Urdu' },
];

export default function BulkUpload() {
  const [file, setFile] = useState(null);
  const [batchId, setBatchId] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [excelPreview, setExcelPreview] = useState(null);
  const [activeBatches, setActiveBatches] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [stopDialog, setStopDialog] = useState({ open: false, batchId: null });
  const [deleteDialog, setDeleteDialog] = useState({ open: false, batchId: null });
  const [isLoading, setIsLoading] = useState(false);

  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const [targetLanguage, setTargetLanguage] = useState('en');
  const [list, setList] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Ref to track if component is mounted
  const isMounted = useRef(true);
  const pollingIntervalRef = useRef(null);

  const showSnackbarMessage = (message, severity = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  // Load active batches from localStorage on component mount
  useEffect(() => {
    isMounted.current = true;
    loadActiveBatches();
    
    // Check for any active processing batches
    checkActiveBatches();
    
    return () => {
      isMounted.current = false;
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Load batches from localStorage
  const loadActiveBatches = () => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const batches = JSON.parse(stored);
        setActiveBatches(batches);
        
        // If there's an active batch, automatically track it
        const activeBatch = batches.find(batch => 
          batch.status === 'processing' || batch.status === 'pending'
        );
        if (activeBatch) {
          setBatchId(activeBatch.batchId);
          setStatus(activeBatch);
          startPolling(activeBatch.batchId);
        }
      }
    } catch (err) {
      console.error('Error loading batches from storage:', err);
    }
  };

  // Save batch to localStorage
  const saveBatchToStorage = (batchData) => {
    try {
      const batches = [...activeBatches];
      const existingIndex = batches.findIndex(b => b.batchId === batchData.batchId);
      
      if (existingIndex >= 0) {
        batches[existingIndex] = batchData;
      } else {
        batches.push(batchData);
      }
      
      setActiveBatches(batches);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(batches));
    } catch (err) {
      console.error('Error saving batch to storage:', err);
    }
  };

  // Remove batch from storage
  const removeBatchFromStorage = (batchIdToRemove) => {
    try {
      const batches = activeBatches.filter(b => b.batchId !== batchIdToRemove);
      setActiveBatches(batches);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(batches));
    } catch (err) {
      console.error('Error removing batch from storage:', err);
    }
  };

  // Check for any active batches that might need polling
  // Updated checkActiveBatches function
const checkActiveBatches = async () => {
  const processingBatches = activeBatches.filter(batch => 
    batch.status === 'processing' || batch.status === 'pending'
  );
  
  // Check each batch against the server
  const batchesToRemove = [];
  
  for (const batch of processingBatches) {
    try {
      const response = await fetch(`${API_BASE_URL}/bulk-status/${batch.batchId}`);
      if (response.ok) {
        const statusData = await response.json();
        saveBatchToStorage({ ...statusData, batchId: batch.batchId });
        
        if (statusData.status === 'processing' || statusData.status === 'pending') {
          if (!batchId) {
            setBatchId(batch.batchId);
            setStatus(statusData);
            startPolling(batch.batchId);
          }
        }
      } else if (response.status === 404) {
        // Mark batch for removal
        batchesToRemove.push(batch.batchId);
        console.log(`Batch ${batch.batchId} not found on server, will remove from localStorage`);
      }
    } catch (err) {
      console.error('Error checking batch status:', err);
    }
  }
  
  // Remove batches that don't exist on server
  batchesToRemove.forEach(batchId => {
    removeBatchFromStorage(batchId);
  });
};

  const handleFileUpload = async (event) => {
  const selectedFile = event.target.files[0];
  if (selectedFile) {
    if (selectedFile.name.endsWith('.xlsx') || selectedFile.name.endsWith('.xls')) {
      setFile(selectedFile);
      setError('');
      
      // ✅ REMOVED: The duplicate API call
      setExcelPreview({
        totalRows: 'Ready for processing',
        message: `File selected: ${selectedFile.name}`
      });
    } else {
      setError('Please upload an Excel file (.xlsx or .xls)');
    }
  }
};

  const startBulkProcessing = async () => {
  if (!file) {
    setError('Please select a file first');
    return;
  }

  // ✅ PREVENT DUPLICATES
  if (isLoading) {
    return;
  }

  setIsLoading(true);
  setLoading(true);
  setError('');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('target_language', targetLanguage);

  try {
    const response = await fetch(`${API_BASE_URL}/bulk-analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start bulk processing');
    }

    const data = await response.json();
    const newBatchId = data.batch_id;
    
    setBatchId(newBatchId);
    
    const initialStatus = {
      batchId: newBatchId,
      status: 'processing',
      total_urls: data.total_urls,
      processed_urls: 0,
      target_language: targetLanguage, 
      failed_urls: 0,
      progress_percentage: 0,
      filename: file.name,
      created_at: new Date().toISOString(),
    };
    
    setStatus(initialStatus);
    saveBatchToStorage(initialStatus);

    startPolling(newBatchId);
    showSnackbarMessage('Bulk processing started successfully!', 'success');
    
  } catch (err) {
    setError(err.message);
    showSnackbarMessage(err.message, 'error');
  } finally {
    setIsLoading(false);
    setLoading(false);
  }
};
  const filteredList = list.filter(item => {
    const meta = item.citnow_metadata || {};
    const searchLower = searchTerm.toLowerCase();

    return (
      (meta.dealership || '').toLowerCase().includes(searchLower) ||
      (meta.vehicle || '').toLowerCase().includes(searchLower) ||
      (meta.service_advisor || '').toLowerCase().includes(searchLower) ||
      (meta.vin || '').toLowerCase().includes(searchLower) ||
      (meta.registration || '').toLowerCase().includes(searchLower) ||
      (meta.email || '').toLowerCase().includes(searchLower) ||
      (meta.phone || '').toLowerCase().includes(searchLower)
    );
  });

 const startPolling = (batchIdToPoll) => {
  if (pollingIntervalRef.current) {
    clearInterval(pollingIntervalRef.current);
  }

  pollingIntervalRef.current = setInterval(async () => {
    if (!isMounted.current) return;

    try {
      const response = await fetch(`${API_BASE_URL}/bulk-status/${batchIdToPoll}`);
      if (response.ok) {
        const statusData = await response.json();
        const updatedStatus = { ...statusData, batchId: batchIdToPoll };
        
        setStatus(updatedStatus);
        saveBatchToStorage(updatedStatus);

        if (['completed', 'failed', 'cancelled'].includes(statusData.status)) {
          clearInterval(pollingIntervalRef.current);
          setLoading(false);
          fetchBatchResults(batchIdToPoll);
        }
      } else if (response.status === 404) {
        // Batch not found - stop polling and remove from storage
        console.warn(`Batch ${batchIdToPoll} not found, stopping polling`);
        clearInterval(pollingIntervalRef.current);
        setLoading(false);
        removeBatchFromStorage(batchIdToPoll);
        
        // Clear current batch if it's the one we're tracking
        if (batchId === batchIdToPoll) {
          setBatchId(null);
          setStatus(null);
        }
        
        showSnackbarMessage('Batch was deleted on server', 'warning');
      }
    } catch (err) {
      console.error('Error polling status:', err);
    }
  }, 5000);
};

  const fetchBatchResults = async (batchId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/bulk-results/${batchId}`);
      const results = await response.json();
      console.log('Batch results:', results);
      
      // Update status with final results if needed
      if (status) {
        const finalStatus = { ...status, ...results };
        saveBatchToStorage(finalStatus);
      }
    } catch (err) {
      console.error('Error fetching batch results:', err);
    }
  };

  // NEW: Stop batch processing
 // NEW: Stop batch processing - FIXED ENDPOINT
const stopBatchProcessing = async (batchIdToStop) => {
  try {
    const response = await fetch(`${API_BASE_URL}/bulk-cancel/${batchIdToStop}`, {
      method: 'POST',
    });

    if (response.ok) {
      showSnackbarMessage('Batch processing stopped successfully', 'success');
      setStopDialog({ open: false, batchId: null });
      
      // Update local status
      if (status && status.batchId === batchIdToStop) {
        const updatedStatus = { ...status, status: 'stopping' };
        setStatus(updatedStatus);
        saveBatchToStorage(updatedStatus);
      }
      
      // Stop polling
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      
      setLoading(false);
    } else {
      throw new Error('Failed to stop batch');
    }
  } catch (err) {
    showSnackbarMessage('Failed to stop batch processing', 'error');
    console.error('Error stopping batch:', err);
  }
};

// NEW: Delete batch and all data - FIXED ENDPOINT
const deleteBatch = async (batchIdToDelete) => {
  try {
    const response = await fetch(`${API_BASE_URL}/bulk-job/${batchIdToDelete}`, {
      method: 'DELETE',
    });

    if (response.ok) {
      showSnackbarMessage('Batch deleted successfully', 'success');
      setDeleteDialog({ open: false, batchId: null });
      
      // Remove from local storage
      removeBatchFromStorage(batchIdToDelete);
      
      // If this is the current batch, clear it
      if (batchId === batchIdToDelete) {
        clearCurrentBatch();
      }
    } else {
      throw new Error('Failed to delete batch');
    }
  } catch (err) {
    showSnackbarMessage('Failed to delete batch', 'error');
    console.error('Error deleting batch:', err);
  }
};
  // NEW: Download results
  const downloadResults = async (batchIdToDownload) => {
    try {
      const response = await fetch(`${API_BASE_URL}/bulk-results/${batchIdToDownload}`);
      const data = await response.json();
      
      // Download as JSON
      const blob = new Blob([JSON.stringify(data.results, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `batch-${batchIdToDownload}-results.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      showSnackbarMessage('Results downloaded successfully', 'success');
    } catch (err) {
      showSnackbarMessage('Failed to download results', 'error');
      console.error('Error downloading results:', err);
    }
  };

  const resumeBatchTracking = (batch) => {
    setBatchId(batch.batchId);
    setStatus(batch);
    
    if (batch.status === 'processing' || batch.status === 'pending') {
      startPolling(batch.batchId);
    }
  };

  const clearCurrentBatch = () => {
    setBatchId(null);
    setStatus(null);
    setFile(null);
    setExcelPreview(null);
    
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'primary';
      case 'pending': return 'warning';
      case 'cancelled': return 'warning';
      case 'stopping': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'failed': return <Error />;
      case 'processing': return <PlayArrow />;
      case 'pending': return <Upload />;
      case 'cancelled': return <Stop />;
      case 'stopping': return <Stop />;
      default: return <Upload />;
    }
  };

  const canStopBatch = (batchStatus) => {
    return ['processing', 'pending'].includes(batchStatus);
  };

  const canDeleteBatch = (batchStatus) => {
    return ['completed', 'failed', 'cancelled', 'stopping'].includes(batchStatus);
  };

  const canDownloadResults = (batchStatus) => {
    return batchStatus === 'completed';
  };

   return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', mt: 4, p: 3 }}>
      <Card elevation={3}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h4" fontWeight="bold">
              📊 Bulk Video Analysis
            </Typography>
            
            <Button
              startIcon={<History />}
              onClick={() => setShowHistory(true)}
              variant="outlined"
            >
              View History ({activeBatches.length})
            </Button>
          </Box>
          
          <Typography variant="body1" color="text.secondary" paragraph>
            Upload your Excel file with Video URLs. The system will automatically detect and process all CitNow videos sequentially.
            <strong> Your progress is saved even if you leave this page!</strong>
          </Typography>

          {/* File Upload Section */}
          <Paper elevation={2} sx={{ p: 3, mb: 3, border: '2px dashed #ccc' }}>
            <Box sx={{ textAlign: 'center' }}>
              <Upload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Upload Excel File
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Your Excel file should contain a <strong>"Video URL"</strong> column with CitNow URLs
              </Typography>
              
              <Button
                variant="outlined"
                component="label"
                sx={{ mb: 2 }}
                disabled={!!batchId && canStopBatch(status?.status)}
              >
                Choose Excel File
                <input
                  type="file"
                  hidden
                  accept=".xlsx,.xls"
                  onChange={handleFileUpload}
                  disabled={!!batchId && canStopBatch(status?.status)}
                />
              </Button>
              
              {file && (
                <Box>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Selected: <strong>{file.name}</strong>
                  </Typography>
                  {excelPreview && (
                    <Alert severity="info" sx={{ mt: 1 }}>
                      ✅ Detected {excelPreview.totalRows} rows with Video URLs
                    </Alert>
                  )}
                </Box>
              )}
            </Box>
          </Paper>

          {/* NEW: Language Selection Section */}
          <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <Translate sx={{ mr: 1 }} />
              Translation Settings
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  select
                  label="Target Translation Language"
                  value={targetLanguage}
                  onChange={(e) => setTargetLanguage(e.target.value)}
                  helperText="Language for translated output"
                  InputProps={{
                    startAdornment: <Translate sx={{ color: 'text.secondary', mr: 1 }} />,
                  }}
                >
                  {LANGS.filter(l => l.code !== 'auto').map(lang => (
                    <MenuItem key={lang.code} value={lang.code}>
                      {lang.name}
                    </MenuItem>
                  ))}
                </TextField>
              </Grid>
            </Grid>
            
          </Paper>

          {/* Error Display */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {/* Active Batch Warning */}
          {batchId && activeBatches.some(b => b.batchId === batchId && (b.status === 'processing' || b.status === 'pending')) && (
            <Alert severity="info" sx={{ mb: 3 }}>
              🔄 <strong>Active processing detected!</strong> You can navigate away from this page - 
              your progress will be saved. Come back anytime to check status.
            </Alert>
          )}

          {/* Excel Format Info */}
          <Paper elevation={1} sx={{ p: 2, mb: 3, bgcolor: 'grey.50' }}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <TableChart sx={{ mr: 1 }} />
              Expected Excel Format:
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>OEM Code</strong></TableCell>
                    <TableCell><strong>Location</strong></TableCell>
                    <TableCell><strong>Video URL</strong></TableCell>
                    <TableCell><strong>Vehicle ID</strong></TableCell>
                    <TableCell><strong>Customer Name</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>38536</TableCell>
                    <TableCell>Kun Motoren Private Limited</TableCell>
                    <TableCell>https://southasia.citnow.com/...</TableCell>
                    <TableCell>TS09FW4707</TableCell>
                    <TableCell>Bmw</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Start Button */}
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={startBulkProcessing}
              disabled={!file || loading || (!!batchId && canStopBatch(status?.status))}
              sx={{ px: 4, py: 1.5, mr: 2 }}
            >
              {loading ? 'Processing...' : 'Start Bulk Processing'}
            </Button>

            {batchId && (
              <Button
                variant="outlined"
                onClick={clearCurrentBatch}
                disabled={loading}
              >
                Start New Upload
              </Button>
            )}
          </Box>

          {/* Progress Display */}
          {status && (
            <Paper elevation={2} sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Processing Status
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Chip 
                    icon={getStatusIcon(status.status)}
                    label={status.status.toUpperCase()} 
                    color={getStatusColor(status.status)}
                  />
                  
                  {/* Action Buttons */}
                  {canStopBatch(status.status) && (
                    <Tooltip title="Stop Processing">
                      <IconButton 
                        color="error"
                        onClick={() => setStopDialog({ open: true, batchId: status.batchId })}
                      >
                        <Stop />
                      </IconButton>
                    </Tooltip>
                  )}
                  
                  {canDownloadResults(status.status) && (
                    <Tooltip title="Download Results">
                      <IconButton 
                        color="primary"
                        onClick={() => downloadResults(status.batchId)}
                      >
                        <Download />
                      </IconButton>
                    </Tooltip>
                  )}
                  
                  {canDeleteBatch(status.status) && (
                    <Tooltip title="Delete Batch">
                      <IconButton 
                        color="error"
                        onClick={() => setDeleteDialog({ open: true, batchId: status.batchId })}
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  )}
                </Box>
              </Box>
              
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Total URLs
                  </Typography>
                  <Typography variant="h6">
                   {status.total_urls}
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Processed
                  </Typography>
                  <Typography variant="h6" color="primary.main">
                    {status.processed_urls}
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Failed
                  </Typography>
                  <Typography variant="h6" color="error.main">
                    {status.failed_urls}
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">
                    Progress
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {status.progress_percentage}%
                  </Typography>
                </Grid>
              </Grid>

              <Box sx={{ mb: 2 }}>
                <LinearProgress 
                  variant="determinate" 
                  value={status.progress_percentage}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>

              {status.current_url && (
                <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1 }}>
                  🎥 Currently processing: {status.current_url}
                </Typography>
              )}

              {status.status === 'completed' && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  <Typography variant="body1" fontWeight="bold">
                    ✅ Bulk processing completed!
                  </Typography>
                  <Typography variant="body2">
                    Successfully processed: {status.processed_urls} videos | Failed: {status.failed_urls} videos
                  </Typography>
                </Alert>
              )}

              {status.status === 'failed' && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  ❌ Processing failed. Check the server logs for details.
                </Alert>
              )}

              {status.status === 'cancelled' && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  ⚠️ Processing was cancelled by user.
                </Alert>
              )}

              <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                Batch ID: {status.batchId} | You can safely navigate away from this page
              </Typography>
            </Paper>
          )}
        </CardContent>
      </Card>

      {/* Batch History Dialog */}
      <Dialog 
        open={showHistory} 
        onClose={() => setShowHistory(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">📋 Processing History</Typography>
            <IconButton onClick={() => setShowHistory(false)}>
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {activeBatches.length === 0 ? (
            <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              No processing history found
            </Typography>
          ) : (
            <Box>
              {activeBatches.map((batch, index) => (
                <Card key={batch.batchId} sx={{ mb: 2, p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {batch.filename || 'Unknown file'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Batch: {batch.batchId} | {new Date(batch.created_at).toLocaleString()}
                      </Typography>
                      <Typography variant="body2">
                        Processed: {batch.processed_urls || 0} / {batch.total_urls || 0} 
                        {batch.failed_urls > 0 && ` | Failed: ${batch.failed_urls}`}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip 
                        icon={getStatusIcon(batch.status)}
                        label={batch.status.toUpperCase()} 
                        color={getStatusColor(batch.status)}
                        size="small"
                      />
                      
                      {(batch.status === 'processing' || batch.status === 'pending') && (
                        <Button
                          size="small"
                          startIcon={<Refresh />}
                          onClick={() => {
                            resumeBatchTracking(batch);
                            setShowHistory(false);
                          }}
                        >
                          Track
                        </Button>
                      )}
                      
                      {canDownloadResults(batch.status) && (
                        <Tooltip title="Download Results">
                          <IconButton 
                            size="small"
                            onClick={() => downloadResults(batch.batchId)}
                          >
                            <Download />
                          </IconButton>
                        </Tooltip>
                      )}
                      
                      {canDeleteBatch(batch.status) && (
                        <Tooltip title="Delete Batch">
                          <IconButton 
                            size="small" 
                            onClick={() => setDeleteDialog({ open: true, batchId: batch.batchId })}
                            color="error"
                          >
                            <Delete />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Box>
                  </Box>
                </Card>
              ))}
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setShowHistory(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Stop Confirmation Dialog */}
      <Dialog open={stopDialog.open} onClose={() => setStopDialog({ open: false, batchId: null })}>
        <DialogTitle>Stop Batch Processing?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to stop this batch processing? 
            The current progress will be saved, but no new URLs will be processed.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStopDialog({ open: false, batchId: null })}>
            Cancel
          </Button>
          <Button 
            onClick={() => stopBatchProcessing(stopDialog.batchId)} 
            color="error"
            variant="contained"
          >
            Stop Processing
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog.open} onClose={() => setDeleteDialog({ open: false, batchId: null })}>
        <DialogTitle>Delete Batch?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this batch and all its results? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false, batchId: null })}>
            Cancel
          </Button>
          <Button 
            onClick={() => deleteBatch(deleteDialog.batchId)} 
            color="error"
            variant="contained"
          >
            Delete Batch
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
