import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Tooltip,
  IconButton,
  LinearProgress,
  Grid,
  Accordion, AccordionSummary, AccordionDetails,
  List, ListItem, ListItemText, ListItemSecondaryAction,
} from '@mui/material';
import {
  Business,
  DirectionsCar,
  Person,
  Description,
  Delete,
  Videocam,
  Mic,
  ArrowForward,
  Error as ErrorIcon,
  Refresh,
  FileDownload,
  Search,
  Email,
  Phone,
  Vibration,
  VolumeUp,
  VolumeOff,
  Warning,
  CheckCircle,
  Score,
  ExpandMore,
  Visibility,
  ArrowBack,
} from '@mui/icons-material';
import axios from 'axios';

const bulkApi = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 60000,
});

const resultsApi = axios.create({
  baseURL: 'http://localhost:5000/results',
  timeout: 30000,
});

export default function ResultsTable({ onSelect }) {
  const [batches, setBatches] = useState([]);
  const [loadingBatches, setLoadingBatches] = useState(true);
  const [errorBatches, setErrorBatches] = useState(null);

  const [individualResults, setIndividualResults] = useState([]);
  const [loadingIndividualResults, setLoadingIndividualResults] = useState(true);
  const [errorIndividualResults, setErrorIndividualResults] = useState(null);

  const [viewMode, setViewMode] = useState('batches'); // 'batches' or 'individual'

  const [selectedbatch_id, setSelectedbatch_id] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [loadingBatchResults, setLoadingBatchResults] = useState(false);
  const [errorBatchResults, setErrorBatchResults] = useState(null);
  const [groupedBatchResults, setGroupedBatchResults] = useState(null);

  const [exportLoading, setExportLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  // --- Helper functions for rendering visuals ---
  const getShakeLevel = useCallback((videoAnalysis) => videoAnalysis?.shake_level || 'Unknown', []);
  const getShakeColor = useCallback((shakeLevel) => {
    switch (shakeLevel?.toLowerCase()) {
      case 'stable': return 'success';
      case 'slightly shaky': return 'warning';
      case 'shaky': return 'warning';
      case 'very shaky': return 'error';
      case 'extremely shaky': return 'error';
      default: return 'default';
    }
  }, []);
  const getShakeIcon = useCallback((shakeLevel) => {
    switch (shakeLevel?.toLowerCase()) {
      case 'stable': return <CheckCircle fontSize="small" />;
      case 'slightly shaky': return <Warning fontSize="small" />;
      case 'shaky': return <Warning fontSize="small" />;
      case 'very shaky': return <Vibration fontSize="small" />;
      case 'extremely shaky': return <Vibration fontSize="small" />;
      default: return <Vibration fontSize="small" />;
    }
  }, []);

  const getAudioClarityLevel = useCallback((audioAnalysis) => audioAnalysis?.clarity_level || audioAnalysis?.prediction || 'Unknown', []);
  const getAudioClarityColor = useCallback((clarityLevel) => {
    switch (clarityLevel?.toLowerCase()) {
      case 'excellent': return 'success';
      case 'good': return 'primary';
      case 'fair': return 'warning';
      case 'poor': return 'error';
      case 'very poor': return 'error';
      case 'clear': return 'success';
      case 'noisy': return 'error';
      default: return 'default';
    }
  }, []);
  const getAudioIcon = useCallback((clarityLevel) => {
    switch (clarityLevel?.toLowerCase()) {
      case 'excellent':
      case 'good':
      case 'clear':
        return <VolumeUp fontSize="small" />;
      case 'fair':
      case 'poor':
      case 'very poor':
      case 'noisy':
        return <VolumeOff fontSize="small" />;
      default:
        return <Mic fontSize="small" />;
    }
  }, []);

  const countAudioIssues = useCallback((audioAnalysis) => audioAnalysis?.issues?.length || 0, []);
  const countVideoIssues = useCallback((videoAnalysis) => videoAnalysis?.issues?.length || 0, []);

  const getQualityStatus = useCallback((quality) => {
    if (quality === undefined || quality === null) return 'Unknown';
    if (quality >= 85) return 'Excellent';
    if (quality >= 70) return 'Good';
    if (quality >= 50) return 'Fair';
    return 'Poor';
  }, []);

  const getStatusColor = useCallback((quality) => {
    if (quality === undefined || quality === null) return 'default';
    if (quality >= 85) return 'success';
    if (quality >= 70) return 'primary';
    if (quality >= 50) return 'warning';
    return 'error';
  }, []);

  const getBatchStatusColor = useCallback((status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'primary';
      case 'pending': return 'info';
      case 'cancelled': return 'warning';
      case 'stopping': return 'warning';
      default: return 'default';
    }
  }, []);

  const getBatchStatusIcon = useCallback((status) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'failed': return <ErrorIcon />;
      case 'processing': return <Videocam />;
      case 'pending': return <Warning />;
      case 'cancelled': return <ErrorIcon />;
      case 'stopping': return <Warning />;
      default: return <Warning />;
    }
  }, []);

  // --- Dealership Export Function ---
  const exportDealershipToExcel = useCallback((dealershipName, results) => {
    try {
      const rows = results.map(item => {
        const m = item.citnow_metadata || {};
        const v = item.video_analysis || {};
        const a = item.audio_analysis || {};
        const t = item.transcription || item.transcription_analysis || {};
        const s = item.summarization || item.summary_analysis || {};
        const overall = item.overall_quality || {};
        const dt = new Date(item.created_at);

        return {
          'Dealership': m.dealership || '',
          'Vehicle': m.vehicle || m.registration || '',
          'Service Advisor': m.service_advisor || '',
          'VIN': m.vin || '',
          'Email': m.email || '',
          'Phone': m.phone || '',

          'Video Quality Score': v.quality_score || 0,
          'Video Quality Label': v.quality_label || '',
          'Camera Stability': v.shake_level || 'Unknown',
          'Video Issues Count': countVideoIssues(v),

          'Audio Quality Score': a.score || 0,
          'Audio Clarity Level': a.clarity_level || '',
          'Audio Prediction': a.prediction || '',
          'Audio Issues Count': countAudioIssues(a),

          'Overall Quality Score': overall.overall_score || 0,
          'Overall Quality Label': overall.overall_label || '',
          'Video Contribution': overall.breakdown?.video_contribution || 0,
          'Audio Contribution': overall.breakdown?.audio_contribution || 0,

          'Transcription': t.text || t.transcription_text || '',
          'Summary': s.summary || s.text || '',

          'Analysis Date': dt.toLocaleDateString(),
          'Processing Order': item.processing_order || 0,
          'Original URL': item.original_url || '',
        };
      });

      if (!rows.length) {
        alert("No data to export for this dealership");
        return;
      }

      // Create CSV content
      const headers = Object.keys(rows[0]);
      const lines = [
        headers.join(','),
        ...rows.map(r =>
          headers.map(h => {
            let cell = '' + (r[h] ?? '');
            cell = cell.replace(/"/g, '""');
            return (cell.includes(',') || cell.includes('\n') || cell.includes('"'))
              ? `"${cell}"`
              : cell;
          }).join(',')
        )
      ];
      const csv = lines.join('\r\n');

      // Create and download file
      const BOM = '\uFEFF';
      const blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;

      // Sanitize dealership name for filename
      const safeDealershipName = dealershipName.replace(/[^\w\s-]/g, '').replace(/[-\s]+/g, '_');
      link.download = `${safeDealershipName}_analysis_${new Date().toISOString().slice(0, 10)}.csv`;

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

    } catch (err) {
      console.error("Dealership export failed", err);
      alert(`Failed to export ${dealershipName} data`);
    }
  }, [countVideoIssues, countAudioIssues]);

  // --- Core Data Fetching Callbacks ---
  const fetchBatches = useCallback(async () => {
    setLoadingBatches(true);
    setErrorBatches(null);
    try {
      const res = await bulkApi.get("/bulk-batches");
      setBatches(res.data.batches);
    } catch (err) {
      console.error('Fetch batches error:', err);
      setErrorBatches(err.response?.data?.detail || err.message || 'Failed to fetch batch history');
    } finally {
      setLoadingBatches(false);
    }
  }, []);

  const fetchIndividualResults = useCallback(async () => {
    setLoadingIndividualResults(true);
    setErrorIndividualResults(null);
    try {
      const res = await resultsApi.get("/");
      const unbatchedResults = res.data.filter(item => !item.batch_id);
      setIndividualResults(unbatchedResults);
    } catch (err) {
      console.error('Fetch individual results error:', err);
      setErrorIndividualResults(err.response?.data?.detail || err.message || 'Failed to fetch individual results');
    } finally {
      setLoadingIndividualResults(false);
    }
  }, []);

  const groupResultsByDealer = useCallback((results) => {
    const grouped = {};
    results.forEach(result => {
      const dealerName = result.citnow_metadata?.dealership || 'Unknown Dealership';
      const sanitizedDealerName = dealerName.replace(/[^\w\s-]/g, '').trim() || 'Unknown Dealership';

      if (!grouped[sanitizedDealerName]) {
        grouped[sanitizedDealerName] = [];
      }
      grouped[sanitizedDealerName].push(result);
    });
    return grouped;
  }, []);
  
  const handleViewDetails = useCallback((item) => {
    console.log('View details clicked for:', item);
    if (onSelect && typeof onSelect === 'function') {
      console.log('Calling onSelect with:', item);
      onSelect(item);
    } else {
      console.log('onSelect not provided or not a function');
      console.log('Analysis details:', item);
      alert(`Analysis Details:\n\nDealership: ${item.citnow_metadata?.dealership || 'N/A'}\nVehicle: ${item.citnow_metadata?.vehicle || 'N/A'}\nOverall Score: ${item.overall_quality?.overall_score || 0}/10\n\nCheck console for full details.`);
    }
  }, [onSelect]);

  const fetchBatchResults = useCallback(async (batch_id) => {
    setLoadingBatchResults(true);
    setErrorBatchResults(null);
    try {
      const res = await bulkApi.get(`/bulk-results/${batch_id}`);
      setBatchResults(res.data.results);
      setGroupedBatchResults(groupResultsByDealer(res.data.results));
    } catch (err)
 {
      console.error(`Fetch results for batch ${batch_id} error:`, err);
      setErrorBatchResults(err.response?.data?.detail || err.message || `Failed to fetch results for batch ${batch_id}`);
    } finally {
      setLoadingBatchResults(false);
    }
  }, [groupResultsByDealer]);

  // --- Action Callbacks ---
  const deleteIndividualResult = useCallback(async (id, batch_id) => {
    if (!window.confirm("Are you sure you want to delete this individual record?")) return;
    try {
      await resultsApi.delete(`/${id}`);
      alert("Record deleted successfully.");
      if (batch_id) {
        fetchBatchResults(batch_id);
      } else {
        fetchIndividualResults();
      }
    } catch (err) {
      console.error("Delete individual result failed", err);
      alert("Failed to delete record");
    }
  }, [fetchBatchResults, fetchIndividualResults]);

  const deleteBatch = useCallback(async (batch_idToDelete) => {
    if (!window.confirm(`Are you sure you want to delete batch ${batch_idToDelete.substring(0, 6)}... and ALL its results? This cannot be undone.`)) return;
    try {
      await bulkApi.delete(`/bulk-job/${batch_idToDelete}`);
      alert(`Batch ${batch_idToDelete.substring(0, 6)}... deleted successfully.`);
      fetchBatches();
      if (selectedbatch_id === batch_idToDelete) {
        setSelectedbatch_id(null);
        setBatchResults(null);
        setGroupedBatchResults(null);
        setSearchTerm('');
      }
    } catch (err) {
      console.error(`Delete batch ${batch_idToDelete} failed`, err);
      alert(`Failed to delete batch ${batch_idToDelete.substring(0, 6)}...`);
    }
  }, [selectedbatch_id, fetchBatches]);

  // --- Effects ---
  useEffect(() => {
    if (viewMode === 'batches') {
      fetchBatches();
    } else {
      fetchIndividualResults();
    }
  }, [viewMode, fetchBatches, fetchIndividualResults]);

  useEffect(() => {
    if (selectedbatch_id) {
      fetchBatchResults(selectedbatch_id);
    }
  }, [selectedbatch_id, fetchBatchResults]);

  // --- Memoized Filtering Logic ---
  const filteredBatches = useMemo(() => {
    const searchLower = searchTerm.toLowerCase();
    return batches.filter(batch => {
      return (
        (batch.filename || '').toLowerCase().includes(searchLower) ||
        (batch.batch_id || '').toLowerCase().includes(searchLower) ||
        (batch.target_language || '').toLowerCase().includes(searchLower) ||
        (batch.status || '').toLowerCase().includes(searchLower)
      );
    });
  }, [batches, searchTerm]);

  const filteredIndividualResults = useMemo(() => {
    const searchLower = searchTerm.toLowerCase();
    return individualResults.filter(item => {
      const meta = item.citnow_metadata || {};
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
  }, [individualResults, searchTerm]);

  const filteredBatchResultsByDealer = useMemo(() => {
    if (!groupedBatchResults) return null;
    const searchLower = searchTerm.toLowerCase();
    const newGrouped = {};

    Object.entries(groupedBatchResults).forEach(([dealerName, results]) => {
      const filteredDealerResults = results.filter(item => {
        const meta = item.citnow_metadata || {};
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
      if (filteredDealerResults.length > 0) {
        newGrouped[dealerName] = filteredDealerResults;
      }
    });
    return newGrouped;
  }, [groupedBatchResults, searchTerm]);

  // --- Export Logic ---
  const exportToExcel = useCallback(async () => {
    setExportLoading(true);
    try {
      let dataToExport = [];
      let exportFileNameSuffix = '';

      if (selectedbatch_id) {
        dataToExport = batchResults || [];
        exportFileNameSuffix = `batch-${selectedbatch_id.substring(0, 6)}`;
      } else if (viewMode === 'individual') {
        dataToExport = filteredIndividualResults;
        exportFileNameSuffix = 'individual-results';
      } else {
        alert("Please select a batch or switch to individual results view to export detailed data.");
        setExportLoading(false);
        return;
      }

      if (!dataToExport || !dataToExport.length) {
        alert("No data to export for the current view/filters.");
        setExportLoading(false);
        return;
      }

      const rows = dataToExport.map(item => {
        const m = item.citnow_metadata || {};
        const v = item.video_analysis || {};
        const a = item.audio_analysis || {};
        const t = item.transcription || {};
        const s = item.summarization || {};
        const x = item.translation || {};
        const dt = new Date(item.created_at);

        return {
          'Batch ID': item.batch_id ? item.batch_id.substring(0, 6) : 'N/A',
          Dealership: m.dealership || '',
          Vehicle: m.vehicle || m.registration || '',
          'Service Advisor': m.service_advisor || '',
          VIN: m.vin || '',
          Email: m.email || '',
          Phone: m.phone || '',

          'Video Quality Score': (v.quality_score || 0).toFixed(1),
          'Video Quality Label': v.quality_label || '',
          'Camera Stability': v.shake_level || 'Unknown',
          'Video Issues Count': countVideoIssues(v),

          'Audio Quality Score': (a.score || 0).toFixed(1),
          'Audio Quality Label': a.prediction || '',
          'Audio Clarity': a.clarity_level || 'Unknown',
          'Audio Issues Count': countAudioIssues(a),

          'Overall Quality Score': (item.overall_quality?.overall_score || 0).toFixed(1),
          'Overall Quality Label': item.overall_quality?.overall_label || '',
          'Video Contribution to Overall': item.overall_quality?.breakdown?.video_contribution || 0,
          'Audio Contribution to Overall': item.overall_quality?.breakdown?.audio_contribution || 0,

          Transcription: t.text || '',
          Summary: s.summary || '',
          Translation: x.translated_text || '',
          'Translation Target Language': x.target_language?.toUpperCase() || '',

          'Processed Date': dt.toLocaleDateString(),
          'Original URL': item.original_url || '',
        };
      });

      const headers = Object.keys(rows[0]);
      const lines = [
        headers.join(','),
        ...rows.map(r =>
          headers.map(h => {
            let cell = '' + (r[h] ?? '');
            cell = cell.replace(/"/g, `""`);
            return (cell.includes(',') || cell.includes('\n') || cell.includes('"'))
              ? `"${cell}"`
              : cell;
          }).join(',')
        )
      ];
      const csv = lines.join('\r\n');

      const BOM = '\uFEFF';
      const blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `citnow-export-${exportFileNameSuffix}-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

    } catch (err) {
      console.error("Export failed", err);
      alert("Failed to export data");
    } finally {
      setExportLoading(false);
    }
  }, [selectedbatch_id, viewMode, batchResults, filteredIndividualResults, countVideoIssues, countAudioIssues]);

  // Individual table row component
  const TableRowWithDetails = ({ item }) => {
    const meta = item.citnow_metadata || {};
    const video = item.video_analysis || {};
    const audio = item.audio_analysis || {};

    const shakeLevel = getShakeLevel(video);
    const audioClarity = getAudioClarityLevel(audio);
    const videoIssuesCount = countVideoIssues(video);
    const audioIssuesCount = countAudioIssues(audio);

    return (
      <TableRow
        hover
        sx={{
          '&:last-child td, &:last-child th': { border: 0 },
          '&:hover': { backgroundColor: 'action.hover' },
          transition: 'all 0.2s ease-in-out',
        }}
      >
        <TableCell sx={{ py: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Business sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
            <Box>
              <Typography variant="body2" fontWeight={500}>
                {meta.dealership || 'N/A'}
              </Typography>
              {meta.email && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                  <Email sx={{ fontSize: 12, mr: 0.5 }} />
                  {meta.email}
                </Typography>
              )}
            </Box>
          </Box>
        </TableCell>

        <TableCell sx={{ py: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <DirectionsCar sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
            <Box>
              <Typography variant="body2" fontWeight={500}>
                {meta.vehicle || meta.registration || 'N/A'}
              </Typography>
              {meta.phone && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                  <Phone sx={{ fontSize: 12, mr: 0.5 }} />
                  {meta.phone}
                </Typography>
              )}
            </Box>
          </Box>
        </TableCell>

        <TableCell sx={{ py: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Person sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
            <Typography variant="body2" fontWeight={500}>
              {meta.service_advisor || 'N/A'}
            </Typography>
          </Box>
        </TableCell>

        <TableCell sx={{ py: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Description sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
            <Typography
              variant="body2"
              fontWeight={500}
              sx={{
                fontFamily: 'monospace',
                backgroundColor: 'grey.100',
                px: 1,
                py: 0.5,
                borderRadius: 1,
                fontSize: '0.75rem',
              }}
            >
              {meta.vin || 'N/A'}
            </Typography>
          </Box>
        </TableCell>

        {/* Video Quality with Shake Indicator */}
        <TableCell sx={{ py: 2 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Tooltip title={`${getQualityStatus(video.quality_score)} - ${video.quality_score || 0}/100`} arrow>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Videocam sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
                <Chip
                  label={`${video.quality_score || 0}/100`}
                  color={getStatusColor(video.quality_score)}
                  size="small"
                  variant="filled"
                />
                {videoIssuesCount > 0 && (
                  <Chip
                    label={videoIssuesCount}
                    color="error"
                    size="small"
                    sx={{ ml: 0.5, height: 20, minWidth: 20 }}
                  />
                )}
              </Box>
            </Tooltip>

            <Tooltip title={`Camera Stability: ${shakeLevel}`} arrow>
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 3 }}>
                {getShakeIcon(shakeLevel)}
                <Chip
                  label={shakeLevel}
                  color={getShakeColor(shakeLevel)}
                  size="small"
                  variant="outlined"
                  sx={{ ml: 0.5, height: 20, fontSize: '0.7rem' }}
                />
              </Box>
            </Tooltip>
          </Box>
        </TableCell>

        {/* Audio Quality with Detailed Indicators */}
        <TableCell sx={{ py: 2 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Tooltip title={`${audioClarity} - ${audio.score || 0}/100`} arrow>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Mic sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
                <Chip
                  label={`${Math.round(audio.score) || 0}/100`}
                  color={getAudioClarityColor(audioClarity)}
                  size="small"
                  variant="filled"
                />
                {audioIssuesCount > 0 && (
                  <Chip
                    label={audioIssuesCount}
                    color="error"
                    size="small"
                    sx={{ ml: 0.5, height: 20, minWidth: 20 }}
                  />
                )}
              </Box>
            </Tooltip>

            <Tooltip title={`Audio Clarity: ${audioClarity}`} arrow>
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 3 }}>
                {getAudioIcon(audioClarity)}
                <Chip
                  label={audioClarity}
                  color={getAudioClarityColor(audioClarity)}
                  size="small"
                  variant="outlined"
                  sx={{ ml: 0.5, height: 20, fontSize: '0.7rem' }}
                />
              </Box>
            </Tooltip>
          </Box>
        </TableCell>

        {/* Overall Quality Column */}
        <TableCell sx={{ py: 2 }}>
          <Tooltip title={`Overall: ${item.overall_quality?.overall_label || 'Unknown'} - ${item.overall_quality?.overall_score || 0}/10`} arrow>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Score sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
              <Chip
                label={`${item.overall_quality?.overall_score?.toFixed(1) || 0}/10`}
                color={
                  (item.overall_quality?.overall_score || 0) >= 8 ? 'success' :
                    (item.overall_quality?.overall_score || 0) >= 6 ? 'primary' :
                      (item.overall_quality?.overall_score || 0) >= 4 ? 'warning' : 'error'
                }
                size="small"
                variant="filled"
                sx={{ fontWeight: 600 }}
              />
            </Box>
          </Tooltip>
          {item.overall_quality?.overall_label && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center', mt: 0.5 }}>
              {item.overall_quality.overall_label}
            </Typography>
          )}
        </TableCell>

        <TableCell sx={{ py: 2 }}>
          <Typography variant="body2" color="text.secondary" fontWeight={500}>
            {new Date(item.created_at).toLocaleDateString()}
          </Typography>
        </TableCell>

        <TableCell sx={{ py: 2 }} align="center">
          <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
            <Tooltip title="View detailed analysis with transcription and summary" arrow>
              <Button
                onClick={() => handleViewDetails(item)}
                endIcon={<ArrowForward />}
                variant="outlined"
                size="small"
                sx={{
                  color: 'primary.main',
                  borderColor: 'primary.main',
                  fontWeight: 600,
                  '&:hover': {
                    backgroundColor: 'primary.main',
                    color: 'white',
                  },
                }}
              >
                Details
              </Button>
            </Tooltip>
            <Tooltip title="Delete this record" arrow>
              <IconButton
                size="small"
                onClick={() => deleteIndividualResult(item._id, item.batch_id)}
                sx={{ color: 'error.main' }}
              >
                <Delete fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </TableCell>
      </TableRow>
    );
  };

  // --- Main Render Logic ---
  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', mt: 4, p: 2 }}>
      <Card elevation={2} sx={{ borderRadius: 2 }}>
        <CardContent sx={{ p: 0 }}>
          {/* Header Section */}
          <Box
            sx={{
              background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
              color: 'white',
              p: 3,
              borderTopLeftRadius: 8,
              borderTopRightRadius: 8,
            }}
          >
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={6}>
                <Typography variant="h4" fontWeight={700} gutterBottom>
                  {selectedbatch_id ? `Batch Results: ${selectedbatch_id.substring(0, 6)}...` : 'Analysis History'}
                </Typography>
                <Typography variant="body1" sx={{ opacity: 0.9 }}>
                  {selectedbatch_id ?
                    `Viewing detailed results for batch ${selectedbatch_id.substring(0, 6)}...` :
                    'Overview of all bulk and individual analysis jobs.'
                  }
                </Typography>
              </Grid>
              <Grid item xs={12} md={6} sx={{ textAlign: { md: 'right' } }}>
                {/* Back to Batches Button */}
                {selectedbatch_id && (
                  <Button
                    onClick={() => {
                      setSelectedbatch_id(null);
                      setBatchResults(null);
                      setGroupedBatchResults(null);
                      setSearchTerm('');
                    }}
                    startIcon={<ArrowBack />}
                    variant="outlined"
                    sx={{
                      backgroundColor: 'white',
                      color: '#1e40af',
                      fontWeight: 600,
                      px: 3,
                      py: 1,
                      mr: 2,
                      '&:hover': {
                        backgroundColor: 'grey.100',
                        transform: 'translateY(-1px)',
                        boxShadow: 2,
                      },
                    }}
                  >
                    Back to Batches
                  </Button>
                )}

                {/* Toggle View Mode Button */}
                {!selectedbatch_id && (
                  <Button
                    onClick={() => setViewMode(prevMode => prevMode === 'batches' ? 'individual' : 'batches')}
                    variant="outlined"
                    startIcon={viewMode === 'batches' ? <DirectionsCar /> : <Business />}
                    sx={{
                      backgroundColor: 'white',
                      color: '#1e40af',
                      fontWeight: 600,
                      px: 3,
                      py: 1,
                      mr: 2,
                      '&:hover': {
                        backgroundColor: 'grey.100',
                        transform: 'translateY(-1px)',
                        boxShadow: 2,
                      },
                    }}
                  >
                    View {viewMode === 'batches' ? 'Individual Results' : 'Batches'}
                  </Button>
                )}

                <Button
                  onClick={exportToExcel}
                  disabled={exportLoading || (selectedbatch_id ? !filteredBatchResultsByDealer || Object.keys(filteredBatchResultsByDealer).length === 0 : (viewMode === 'batches' ? !filteredBatches.length : !filteredIndividualResults.length))}
                  startIcon={exportLoading ? <CircularProgress size={16} color="inherit" /> : <FileDownload />}
                  variant="contained"
                  sx={{
                    backgroundColor: 'white',
                    color: '#1e40af',
                    fontWeight: 600,
                    px: 3,
                    py: 1,
                    '&:hover': {
                      backgroundColor: 'grey.100',
                      transform: 'translateY(-1px)',
                      boxShadow: 2,
                    },
                    '&:disabled': {
                      backgroundColor: 'grey.300',
                    },
                    mr: 2,
                    mb: { xs: 2, md: 0 },
                  }}
                >
                  {exportLoading ? 'Exporting...' : (selectedbatch_id ? 'Export Batch Excel' : (viewMode === 'batches' ? 'Export Batches CSV' : 'Export Individual CSV'))}
                </Button>
                <Button
                  onClick={selectedbatch_id ? () => fetchBatchResults(selectedbatch_id) : (viewMode === 'batches' ? fetchBatches : fetchIndividualResults)}
                  startIcon={<Refresh />}
                  variant="outlined"
                  sx={{
                    color: 'white',
                    borderColor: 'white',
                    '&:hover': {
                      backgroundColor: 'rgba(255,255,255,0.1)',
                      borderColor: 'white',
                    },
                  }}
                >
                  Refresh {selectedbatch_id ? 'Results' : (viewMode === 'batches' ? 'Batches' : 'Individual')}
                </Button>
              </Grid>
            </Grid>
          </Box>

          {/* Search Bar */}
          <Box sx={{ backgroundColor: 'grey.50', p: 2, borderBottom: 1, borderColor: 'grey.200' }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Search sx={{ color: 'text.secondary' }} />
                  <input
                    type="text"
                    placeholder={selectedbatch_id ? "Search dealer, vehicle, advisor..." : (viewMode === 'batches' ? "Search batch ID, filename, status..." : "Search dealership, vehicle, advisor...")}
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{
                      border: 'none',
                      outline: 'none',
                      background: 'transparent',
                      width: '100%',
                      fontSize: '14px',
                      padding: '8px 0',
                    }}
                  />
                </Box>
              </Grid>
              <Grid item xs={12} md={6} sx={{ textAlign: { md: 'right' } }}>
                <Typography variant="body2" color="text.secondary">
                  {selectedbatch_id
                    ? `Showing ${Object.values(filteredBatchResultsByDealer || {}).flat().length || 0} of ${batchResults?.length || 0} results`
                    : (viewMode === 'batches'
                      ? `Showing ${filteredBatches.length} of ${batches.length} batches`
                      : `Showing ${filteredIndividualResults.length} of ${individualResults.length} individual results`)}
                </Typography>
              </Grid>
            </Grid>
          </Box>

          {/* Conditional Content Area */}
          <Box sx={{ p: 2 }}>
            {selectedbatch_id ? (
              // Viewing results for a specific batch (dealer-wise)
              loadingBatchResults ? (
                <Box sx={{ width: '100%', minHeight: 200, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                  <CircularProgress size={40} thickness={3} />
                  <Typography variant="body1" sx={{ mt: 1, color: 'text.secondary' }}>
                    Loading detailed batch results...
                  </Typography>
                </Box>
              ) : errorBatchResults ? (
                <Alert severity="error" action={<Button color="inherit" size="small" onClick={() => fetchBatchResults(selectedbatch_id)} startIcon={<Refresh />}>Retry</Button>}>
                  {errorBatchResults}
                </Alert>
              ) : filteredBatchResultsByDealer && Object.keys(filteredBatchResultsByDealer).length > 0 ? (
                <Box>
                  {Object.entries(filteredBatchResultsByDealer).sort(([dealerA], [dealerB]) => dealerA.localeCompare(dealerB)).map(([dealerName, results]) => (
                    <Accordion key={dealerName} defaultExpanded sx={{ mb: 1 }}>
                      <AccordionSummary
                        expandIcon={<ExpandMore />}
                        aria-controls={`${dealerName}-content`}
                        id={`${dealerName}-header`}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                          <Typography variant="h6" sx={{ flexGrow: 1 }}>
                            {dealerName} ({results.length} videos)
                          </Typography>
                          <Tooltip title={`Export ${dealerName} to Excel`}>
                            <IconButton
                              onClick={(e) => {
                                e.stopPropagation();
                                exportDealershipToExcel(dealerName, results);
                              }}
                              color="primary"
                              size="small"
                              sx={{ ml: 1 }}
                            >
                              <FileDownload />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails sx={{ p: 0 }}>
                        <List disablePadding>
                          {results.map((result) => (
                            <ListItem
                              key={result._id}
                              secondaryAction={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  {result.overall_quality?.overall_score !== undefined && (
                                    <Chip
                                      icon={<Score fontSize="small" />}
                                      label={`${result.overall_quality.overall_score.toFixed(1)}/10`}
                                      color={
                                        result.overall_quality.overall_score >= 8 ? 'success' :
                                        result.overall_quality.overall_score >= 6 ? 'primary' :
                                        result.overall_quality.overall_score >= 4 ? 'warning' : 'error'
                                      }
                                      size="small"
                                      sx={{ fontWeight: 600 }}
                                    />
                                  )}
                                  {/* ======================= FIX IS HERE ======================= */}
                                  <Button
                                    variant="outlined"
                                    size="small"
                                    startIcon={<Visibility />}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleViewDetails(result); // Use the existing handler
                                    }}
                                    sx={{ ml: 1 }}
                                  >
                                    View
                                  </Button>
                                  {/* ========================================================== */}
                                  <Tooltip title="Delete this record">
                                    <IconButton 
                                      edge="end" 
                                      aria-label="delete" 
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        deleteIndividualResult(result._id, result.batch_id);
                                      }}
                                    >
                                      <Delete fontSize="small" color="error" />
                                    </IconButton>
                                  </Tooltip>
                                </Box>
                              }
                              divider
                            >
                              <ListItemText
                                primary={
                                  <Typography variant="body1" component="span" sx={{ fontWeight: 'medium' }}>
                                    {result.citnow_metadata?.vehicle || result.citnow_metadata?.registration || `Video ID: ${result._id.substring(0, 6)}...`}
                                  </Typography>
                                }
                                secondary={
                                  <Box>
                                    <Typography variant="body2">
                                      Reg: {result.citnow_metadata?.registration || 'N/A'} | VIN: {result.citnow_metadata?.vin || 'N/A'}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                      Processed: {new Date(result.created_at).toLocaleDateString()}
                                    </Typography>
                                  </Box>
                                }
                              />
                            </ListItem>
                          ))}
                        </List>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                  {(!filteredBatchResultsByDealer || Object.keys(filteredBatchResultsByDealer).length === 0) && searchTerm && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      No results found matching your search across all dealers in this batch.
                    </Alert>
                  )}
                  {(!filteredBatchResultsByDealer || Object.keys(filteredBatchResultsByDealer).length === 0) && !searchTerm && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      No detailed results available for this batch. It might be empty or still processing.
                    </Alert>
                  )}
                </Box>
              ) : (
                <Alert severity="info">
                  No detailed results available for this batch. It might be empty or still processing.
                </Alert>
              )
            ) : (
              // Not viewing a specific batch: display list of all batches or individual results
              viewMode === 'batches' ? (
                // Display List of Batches
                loadingBatches ? (
                  <Box sx={{ width: '100%', minHeight: 400, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                    <CircularProgress size={60} thickness={4} />
                    <Typography variant="h6" sx={{ mt: 2, color: 'text.secondary' }}>
                      Loading batch history...
                    </Typography>
                    <LinearProgress sx={{ width: '80%', mt: 2, maxWidth: 400 }} />
                  </Box>
                ) : errorBatches ? (
                  <Alert severity="error" action={<Button color="inherit" size="small" onClick={fetchBatches} startIcon={<Refresh />}>Retry</Button>}>
                    {errorBatches}
                  </Alert>
                ) : (
                  <TableContainer
                    component={Paper}
                    elevation={0}
                    sx={{
                      maxHeight: 600,
                      '&::-webkit-scrollbar': { width: 8 },
                      '&::-webkit-scrollbar-track': { background: '#f1f1f1' },
                      '&::-webkit-scrollbar-thumb': { background: '#c1c1c1', borderRadius: 4 },
                    }}
                  >
                    <Table stickyHeader>
                      <TableHead>
                        <TableRow sx={{ backgroundColor: 'grey.50' }}>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Batch ID</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Filename</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Status</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Progress</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Processed/Total</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Target Language</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Created At</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }} align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filteredBatches.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={8} align="center" sx={{ py: 6 }}>
                              <Box sx={{ textAlign: 'center' }}>
                                <Search sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                                <Typography variant="h6" color="text.secondary" gutterBottom>
                                  No batches found
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  {searchTerm ? 'Try adjusting your search terms' : 'No bulk analysis batches have been run yet.'}
                                </Typography>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ) : (
                          filteredBatches.map((batch) => (
                            <TableRow key={batch.batch_id} hover>
                              <TableCell>{batch.batch_id ? batch.batch_id.substring(0, 6) + '...' : 'N/A'}</TableCell>
                              <TableCell>{batch.filename}</TableCell>
                              <TableCell>
                                <Chip
                                  icon={getBatchStatusIcon(batch.status)}
                                  label={batch.status.toUpperCase()}
                                  color={getBatchStatusColor(batch.status)}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <LinearProgress variant="determinate" value={batch.progress_percentage || 0} sx={{ height: 6, borderRadius: 3 }} />
                                <Typography variant="caption" color="text.secondary">{batch.progress_percentage?.toFixed(1) || 0}%</Typography>
                              </TableCell>
                              <TableCell>{batch.processed_urls}/{batch.total_urls}</TableCell>
                              <TableCell>{batch.target_language?.toUpperCase() || 'EN'}</TableCell>
                              <TableCell>{new Date(batch.created_at).toLocaleDateString()}</TableCell>
                              <TableCell align="center">
                                <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
                                  <Tooltip title="View Batch Results">
                                    <IconButton
                                      size="small"
                                      onClick={() => setSelectedbatch_id(batch.batch_id)}
                                      color="primary"
                                      disabled={['pending', 'processing', 'stopping'].includes(batch.status)}
                                    >
                                      <ArrowForward fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="Delete Entire Batch">
                                    <IconButton
                                      size="small"
                                      onClick={() => deleteBatch(batch.batch_id)}
                                      color="error"
                                      disabled={['pending', 'processing', 'stopping'].includes(batch.status)}
                                    >
                                      <Delete fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                </Box>
                              </TableCell>
                            </TableRow>
                          ))
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )
              ) : (
                // CORRECTED: Display List of Individual Results
                loadingIndividualResults ? (
                  <Box sx={{ width: '100%', minHeight: 400, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                    <CircularProgress size={60} thickness={4} />
                    <Typography variant="h6" sx={{ mt: 2, color: 'text.secondary' }}>
                      Loading individual analyses...
                    </Typography>
                    <LinearProgress sx={{ width: '80%', mt: 2, maxWidth: 400 }} />
                  </Box>
                ) : errorIndividualResults ? (
                  <Alert severity="error" action={<Button color="inherit" size="small" onClick={fetchIndividualResults} startIcon={<Refresh />}>Retry</Button>}>
                    {errorIndividualResults}
                  </Alert>
                ) : (
                  <TableContainer
                    component={Paper}
                    elevation={0}
                    sx={{
                      maxHeight: 600,
                      '&::-webkit-scrollbar': { width: 8 },
                      '&::-webkit-scrollbar-track': { background: '#f1f1f1' },
                      '&::-webkit-scrollbar-thumb': { background: '#c1c1c1', borderRadius: 4 },
                    }}
                  >
                    <Table stickyHeader>
                      <TableHead>
                        <TableRow sx={{ backgroundColor: 'grey.50' }}>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Dealership</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Vehicle</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Service Advisor</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>VIN</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Video Quality</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Audio Quality</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }} align="center">Overall Score</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }}>Date</TableCell>
                          <TableCell sx={{ fontWeight: 700, py: 2 }} align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filteredIndividualResults.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={9} align="center" sx={{ py: 6 }}>
                              <Box sx={{ textAlign: 'center' }}>
                                <Search sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                                <Typography variant="h6" color="text.secondary" gutterBottom>
                                  No individual results found
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  {searchTerm ? 'Try adjusting your search terms' : 'No individual (non-batch) analyses have been run yet.'}
                                </Typography>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ) : (
                          filteredIndividualResults.map((item) => (
                            <TableRowWithDetails key={item._id} item={item} />
                          ))
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )
              )
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
