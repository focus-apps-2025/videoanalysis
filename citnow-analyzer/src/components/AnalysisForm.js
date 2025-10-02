import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  MenuItem,
  Button,
  Typography,
  CircularProgress,
  Grid,
  Paper,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  VideoCameraBack,
  Translate,
  Language,
  Analytics,
  Error as ErrorIcon,
} from '@mui/icons-material';
import axios from 'axios';

const LANGS = [
  { code: "auto", name: "Auto Detect" },
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

// Create axios instance with timeout
const api = axios.create({
  baseURL: 'http://localhost:5000/analyze',
 // 30 seconds
});

export default function AnalysisForm({ onAnalyze }) {
  const [url, setUrl] = useState("");
  const [lang, setLang] = useState("auto");
  const [target, setTarget] = useState("ta");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const res = await api.post("http://localhost:5000/analyze", {
        citnow_url: url, 
        transcription_language: lang, 
        target_language: target
      });
      onAnalyze(res.data);
    } catch (err) {
      console.error('API Error:', err);
      
      if (err.code === 'ECONNREFUSED') {
        setError('Backend server is not running. Please start the backend server on port 8000.');
      } else if (err.code === 'NETWORK_ERROR') {
        setError('Network error. Please check your connection and ensure backend is running.');
      } else if (err.response?.status === 404) {
        setError('API endpoint not found. Please check backend routes.');
      } else if (err.response?.status >= 500) {
        setError('Server error. Please check backend logs.');
      } else {
        setError(err.response?.data?.detail || err.message || 'An unexpected error occurred');
      }
    }
    setLoading(false);
  };

  const handleCloseError = () => {
    setError(null);
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4 }}>
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert 
          severity="error" 
          onClose={handleCloseError}
          icon={<ErrorIcon />}
        >
          {error}
        </Alert>
      </Snackbar>

      <Card elevation={3}>
        <CardContent sx={{ p: 4 }}>
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Paper
              elevation={2}
              sx={{
                width: 80,
                height: 80,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mx: 'auto',
                mb: 3,
                background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
              }}
            >
              <VideoCameraBack sx={{ fontSize: 40, color: 'white' }} />
            </Paper>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 700 }}>
              Professional Video Analysis
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Enter CitNow URL for comprehensive service video analysis
            </Typography>
          </Box>

          <form onSubmit={submit}>
            <TextField
              fullWidth
              label="CitNow Video URL"
              placeholder="https://southasia.citnow.com/..."
              value={url}
              onChange={e => setUrl(e.target.value)}
              required
              sx={{ mb: 3 }}
              InputProps={{
                startAdornment: <VideoCameraBack sx={{ color: 'text.secondary', mr: 1 }} />,
              }}
            />

            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  select
                  label="Spoken Language"
                  value={lang}
                  onChange={e => setLang(e.target.value)}
                  InputProps={{
                    startAdornment: <Language sx={{ color: 'text.secondary', mr: 1 }} />,
                  }}
                >
                  {LANGS.map(lang => (
                    <MenuItem key={lang.code} value={lang.code}>
                      {lang.name}
                    </MenuItem>
                  ))}
                </TextField>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  select
                  label="Target Language"
                  value={target}
                  onChange={e => setTarget(e.target.value)}
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

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <Analytics />}
              sx={{
                py: 1.5,
                background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%)',
                  color:'#ffff'
                  
                },
              }}
            >
              {loading ? 'Analyzing Video... this Process take 3 to 5mins' : 'Analyze Video'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </Box>
  );
}