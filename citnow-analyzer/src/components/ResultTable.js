import React, { useEffect, useState, useCallback } from 'react';
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
  useTheme,
  CircularProgress,
  Alert,
  Tooltip,
  IconButton,
  LinearProgress,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Business,
  DirectionsCar,
  Person,
  Description,
  Delete,
  Videocam,
  Mic,
  Visibility,
  ArrowForward,
  Error as ErrorIcon,
  Refresh,
  Download,
  FileDownload,
  Search,
  Email,
  Phone,
  Summarize,
  Translate,
  BugReport,
} from '@mui/icons-material';
import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: 'https://videoanalysis-e55w.onrender.com/results',
  timeout: 30000,
});

export default function ResultsTable({ onSelect }) {
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [exportLoading, setExportLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [debugData, setDebugData] = useState(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const theme = useTheme();
  const deleteItem = async (id) => {
    if (!window.confirm("Are you sure you want to delete this record?")) return;
    try {
      await api.delete(`/${id}`);
      // re‐load the list
      fetchResults();
    } catch (err) {
      console.error("Delete failed", err);
      alert("Failed to delete record");
    }
  };
  const fetchResults = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get("/");
      console.log('API Response:', res.data); // Debug log
      setList(res.data);
    } catch (err) {
      console.error('Fetch error:', err);
      if (err.code === 'ECONNREFUSED') {
        setError('Cannot connect to backend server. Please make sure it is running on port 5000.');
      } else if (err.response?.status === 404) {
        setError('Endpoint not found. Please check the server URL.');
      } else {
        setError(err.response?.data?.detail || err.message || 'Failed to fetch results');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  // Debug function to check data structure
  const debugDataStructure = (item) => {
    setDebugData({
      rawData: item,
      availableKeys: Object.keys(item),
      citnowMetadata: item.citnow_metadata || 'Not found',
      videoAnalysis: item.video_analysis || 'Not found',
      audioAnalysis: item.audio_analysis || 'Not found',
      transcriptionAnalysis: item.transcription_analysis || 'Not found',
      summaryAnalysis: item.summary_analysis || 'Not found',
      translationAnalysis: item.translation_analysis || 'Not found',
    });
    setDebugOpen(true);
  };

  // Enhanced Excel export function with better data extraction
const exportToExcel = async () => {
  setExportLoading(true);
  try {
    const rows = list.map(item => {
      const m  = item.citnow_metadata   || {};
      const v  = item.video_analysis    || {};
      const a  = item.audio_analysis    || {};
      const t  = item.transcription      || item.transcription_analysis || {};
      const s  = item.summarization      || item.summary_analysis        || {};
      const x  = item.translation        || item.translation_analysis    || {};
      const dt = new Date(item.created_at);

      return {
        Dealership:        m.dealership    || '',
        Vehicle:           m.vehicle || m.registration || '',
        'Service Advisor': m.service_advisor || '',
        VIN:               m.vin            || '',
        Email:             m.email          || '',
        Phone:             m.phone          || '',

        'Video Quality':   `${(v.quality_score||0).toFixed(1)} ${v.quality_label||''}`.trim(),
        'Audio Quality':   `${(a.score||0).toFixed(1)} ${a.prediction||''}`.trim(),

        Transcription:     t.text || t.transcription_text || '',
        Summary:           s.summary || s.text          || '',
        Translation:       x.translated_text || x.text || '',

        Date:              dt.toLocaleDateString(),
       
      };
    });

    if (!rows.length) {
      alert("No data to export");
      return;
    }

    // build CSV
    const headers = Object.keys(rows[0]);
    const lines = [
      headers.join(','), 
      ...rows.map(r =>
        headers.map(h => {
          let cell = '' + (r[h] ?? '');
          cell = cell.replace(/"/g, `""`);
          return (cell.includes(',')||cell.includes('\n')) 
            ? `"${cell}"` 
            : cell;
        }).join(',')
      )
    ];
    const csv = lines.join('\r\n');

    // prepend BOM so Excel opens as UTF-8
    const BOM = '\uFEFF';
    const blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `citnow-export-${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

  } catch (err) {
    console.error("Export failed", err);
    alert("Failed to export data");
  } finally {
    setExportLoading(false);
  }
};



  // Helper functions
  const getQualityStatus = (quality) => {
    if (!quality) return 'Unknown';
    if (quality >= 85) return 'Excellent';
    if (quality >= 70) return 'Good';
    if (quality >= 50) return 'Fair';
    return 'Poor';
  };

  const getAudioStatus = (prediction) => {
    if (!prediction) return 'Unknown';
    return prediction === 'Clear' ? 'Good' : 'Noisy';
  };

  const getStatusColor = (quality) => {
    if (!quality) return 'default';
    if (quality >= 85) return 'success';
    if (quality >= 70) return 'primary';
    if (quality >= 50) return 'warning';
    return 'error';
  };

  const getAudioColor = (prediction) => {
    return prediction === 'Clear' ? 'success' : 'error';
  };

  // Filter data based on search term
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

  // Enhanced table row component to show more details
  const TableRowWithDetails = ({ item }) => {
    const meta = item.citnow_metadata || {};
    const video = item.video_analysis || {};
    const audio = item.audio_analysis || {};
    const transcription = item.transcription_analysis || item.transcription || {};

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

        <TableCell sx={{ py: 2 }}>
          <Tooltip title={`${getQualityStatus(video.quality_score)} - ${video.quality_score || 0}/100`} arrow>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Videocam sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
              <Chip
                label={`${video.quality_score || 0}/100`}
                color={getStatusColor(video.quality_score)}
                size="small"
                variant="filled"
              />
            </Box>
          </Tooltip>
        </TableCell>

        <TableCell sx={{ py: 2 }}>
          <Tooltip title={`${getAudioStatus(audio.prediction)} - ${audio.quality_score || 0}/100`} arrow>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Mic sx={{ color: 'primary.main', mr: 1, fontSize: 20 }} />
              <Chip
                label={`${Math.round(audio.score) || 0}/100`}
                color={getAudioColor(audio.prediction)}
                size="small"
                variant="filled"
              />
            </Box>
          </Tooltip>
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
                onClick={() => onSelect(item)}
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
                onClick={() => deleteItem(item._id)}
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

  if (loading) {
    return (
      <Box sx={{ width: '100%', minHeight: 400, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        <CircularProgress size={60} thickness={4} />
        <Typography variant="h6" sx={{ mt: 2, color: 'text.secondary' }}>
          Loading service analyses...
        </Typography>
        <LinearProgress sx={{ width: '80%', mt: 2, maxWidth: 400 }} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4 }}>
        <Alert
          severity="error"
          action={
            <Button color="inherit" size="small" onClick={fetchResults} startIcon={<Refresh />}>
              Retry
            </Button>
          }
          sx={{ mb: 2 }}
        >
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <>
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
                    CitNow Service Analyses
                  </Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>
                    Complete service video evaluations with quality metrics, transcriptions, and summaries
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6} sx={{ textAlign: { md: 'right' } }}>
                  <Button
                    onClick={exportToExcel}
                    disabled={exportLoading || list.length === 0}
                    startIcon={exportLoading ? <CircularProgress size={16} /> : <FileDownload />}
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
                    {exportLoading ? 'Exporting...' : 'Export Full Excel'}
                  </Button>
                  <Button
                    onClick={fetchResults}
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
                    Refresh
                  </Button>
                </Grid>
              </Grid>
            </Box>

            {/* Stats Bar */}
            <Box sx={{ backgroundColor: 'grey.50', p: 2, borderBottom: 1, borderColor: 'grey.200' }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Search sx={{ color: 'text.secondary' }} />
                    <input
                      type="text"
                      placeholder="Search dealership, vehicle, advisor, VIN, email..."
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
                    Showing {filteredList.length} of {list.length} service analyses
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Table Section */}
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
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>Dealership & Email</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>Vehicle & Phone</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>Service Advisor</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>VIN</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>Video Quality</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>Audio Quality</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }}>Date</TableCell>
                    <TableCell sx={{ fontWeight: 700, py: 2 }} align="center">Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredList.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} align="center" sx={{ py: 6 }}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Search sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
                          <Typography variant="h6" color="text.secondary" gutterBottom>
                            No service analyses found
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {searchTerm ? 'Try adjusting your search terms' : 'No data available'}
                          </Typography>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredList.map((item) => (
                      <TableRowWithDetails key={item._id} item={item} />
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Box>

      {/* Debug Dialog */}
      <Dialog open={debugOpen} onClose={() => setDebugOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Data Structure Debug</DialogTitle>
        <DialogContent>
          {debugData && (
            <Box sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
              <Typography variant="h6" gutterBottom>Available Keys:</Typography>
              <pre>{JSON.stringify(debugData.availableKeys, null, 2)}</pre>

              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>CitNow Metadata:</Typography>
              <pre>{JSON.stringify(debugData.citnowMetadata, null, 2)}</pre>

              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Transcription Analysis:</Typography>
              <pre>{JSON.stringify(debugData.transcriptionAnalysis, null, 2)}</pre>

              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Summary Analysis:</Typography>
              <pre>{JSON.stringify(debugData.summaryAnalysis, null, 2)}</pre>

              <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Translation Analysis:</Typography>
              <pre>{JSON.stringify(debugData.translationAnalysis, null, 2)}</pre>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDebugOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
