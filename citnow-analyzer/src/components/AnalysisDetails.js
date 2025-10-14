import React from 'react';
import {
  Box, Card, CardContent, Typography, Button, Grid,
  Paper, Chip, Divider, List, ListItem, ListItemIcon,
  ListItemText, Alert,CircularProgress
} from '@mui/material';
import {
  ArrowBack, PictureAsPdf, DirectionsCar, Person,
  Phone, Email, Videocam, Mic, Description, Translate,
  Business, Summarize, Warning, CheckCircle,
  Vibration, VolumeOff, BlurOn, Grain,Score
} from '@mui/icons-material';
import { generatePDF } from './pdfgenerator';

export default function AnalysisDetails({ analysis: rawResult, onClose: onBack }) {
  if (!rawResult) return null;

  // 1) Also unwrap the `.results` key
  const result =
    rawResult.results  // <-- your real payload
    ?? rawResult.data
    ?? rawResult.result
    ?? rawResult;

  // 2) Normalize sub-objects
  const metadata      = result.citnow_metadata   || {};
  const videoAnalysis = result.video_analysis    || {};
  const audioAnalysis = result.audio_analysis    || {};

  const transcription = result.transcription      || {};
  const summarization = result.summarization      || {};
  const translation   = result.translation        || {};

  // 3) Helper functions for quality indicators
  const getShakeSeverity = (shakeLevel) => {
    switch (shakeLevel?.toLowerCase()) {
      case 'extremely shaky': return { color: 'error', severity: 'high' };
      case 'very shaky': return { color: 'error', severity: 'high' };
      case 'shaky': return { color: 'warning', severity: 'medium' };
      case 'slightly shaky': return { color: 'warning', severity: 'low' };
      case 'stable': return { color: 'success', severity: 'none' };
      default: return { color: 'default', severity: 'unknown' };
    }
  };

  const getAudioIssueSeverity = (clarityLevel) => {
    switch (clarityLevel?.toLowerCase()) {
      case 'very poor': return { color: 'error', severity: 'high' };
      case 'poor': return { color: 'error', severity: 'high' };
      case 'fair': return { color: 'warning', severity: 'medium' };
      case 'good': return { color: 'success', severity: 'low' };
      case 'excellent': return { color: 'success', severity: 'none' };
      default: return { color: 'default', severity: 'unknown' };
    }
  };

  // 4) Re-usable UI pieces
  const InfoCard = ({ label, value, icon }) => (
    <Paper elevation={1} sx={{ p:2, height:'100%' }}>
      <Box sx={{ display:'flex', alignItems:'center', mb:1 }}>
        <Box sx={{ color:'primary.main', mr:1 }}>{icon}</Box>
        <Typography variant="subtitle2" color="text.secondary">{label}</Typography>
      </Box>
      <Typography variant="body1" fontWeight={500}>{value}</Typography>
    </Paper>
  );

  const QualityCard = ({ title, score, label, icon, subtitle, issues = [] }) => (
    <Paper elevation={2} sx={{
      p:3,
      background:'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
      border:'1px solid grey.100'
    }}>
      <Box sx={{ display:'flex', alignItems:'center', mb:2 }}>
        <Box sx={{ p:1, backgroundColor:'primary.light', borderRadius:2, mr:2, color:'white' }}>
          {icon}
        </Box>
        <Box>
          <Typography variant="h6" fontWeight={600}>{title}</Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>
      </Box>
      <Box sx={{ display:'flex', alignItems:'baseline', mb:1 }}>
        <Typography variant="h3" fontWeight={700} color="primary.main">
          {score ?? 0}
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ ml:1 }}>/100</Typography>
      </Box>
      <Chip
        label={label || 'N/A'}
        color={
          label === 'Excellent' ? 'success' :
          label === 'Good'      ? 'primary' :
          label === 'Fair'      ? 'warning' :
                                 'error'
        }
        sx={{ mb: issues.length > 0 ? 2 : 0 }}
      />

      {/* Display issues if any */}
      {issues.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Detected Issues:
          </Typography>
          <List dense sx={{ py: 0 }}>
            {issues.slice(0, 3).map((issue, index) => (
              <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Warning fontSize="small" color="warning" />
                </ListItemIcon>
                <ListItemText primary={issue} />
              </ListItem>
            ))}
            {issues.length > 5 && (
              <ListItem sx={{ py: 0.5, px: 0 }}>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Warning fontSize="small" color="warning" />
                </ListItemIcon>
                <ListItemText primary={`+${issues.length - 3} more issues`} />
              </ListItem>
            )}
          </List>
        </Box>
      )}
    </Paper>
  );

  const ShakeIndicator = ({ shakeLevel, videoAnalysis }) => {
    const { color, severity } = getShakeSeverity(shakeLevel);
    
    // Get stability score from component_scores (it's already a number 0-100)
    const stabilityScore = videoAnalysis?.component_scores?.stability;
    
    return (
        <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Vibration sx={{ color: `${color}.main`, mr: 1 }} />
                <Typography variant="h6" fontWeight={600}>
                    Camera Stability
                </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                    <Chip 
                        label={shakeLevel || 'Unknown'} 
                        color={color}
                        variant={severity === 'none' ? 'outlined' : 'filled'}
                    />
                    {stabilityScore !== undefined && stabilityScore !== null && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            Stability score: {stabilityScore}%
                        </Typography>
                    )}
                </Box>
                {severity !== 'none' && severity !== 'unknown' && (
                    <Alert severity={color} sx={{ flex: 1, ml: 2 }}>
                        {severity === 'high' && 'Significant camera shake detected - affects viewing experience'}
                        {severity === 'medium' && 'Noticeable camera shake present'}
                        {severity === 'low' && 'Minor camera shake detected'}
                    </Alert>
                )}
            </Box>
        </Paper>
    );
};

  const AudioQualityDetails = ({ audioAnalysis }) => {
    const { color, severity } = getAudioIssueSeverity(audioAnalysis.clarity_level);
    
    return (
      <Paper elevation={1} sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Mic sx={{ color: `${color}.main`, mr: 1 }} />
          <Typography variant="h6" fontWeight={600}>
            Audio Quality Details
          </Typography>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Clarity Level
              </Typography>
              <Chip 
                label={audioAnalysis.clarity_level || 'Unknown'} 
                color={color}
                sx={{ mt: 0.5 }}
              />
            </Box>
            
            {audioAnalysis.detailed_analysis && (
              <Box>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Technical Analysis:
                </Typography>
                <List dense>
                  <ListItem sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <VolumeOff fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={`Volume: ${audioAnalysis.detailed_analysis.volume_level || 'N/A'}`} 
                    />
                  </ListItem>
                  <ListItem sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <Grain fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={`Noise: ${audioAnalysis.detailed_analysis.noise_level || 'N/A'}`} 
                    />
                  </ListItem>
                </List>
              </Box>
            )}
          </Grid>
          
          <Grid item xs={12} md={6}>
            {audioAnalysis.issues && audioAnalysis.issues.length > 0 && (
              <Box>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Audio Issues:
                </Typography>
                <List dense>
                  {audioAnalysis.issues.map((issue, index) => (
                    <ListItem key={index} sx={{ px: 0 }}>
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        <Warning fontSize="small" color="warning" />
                      </ListItemIcon>
                      <ListItemText primary={issue} />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>
    );
  };

  const ContentSection = ({ title, content, icon }) => (
    <Paper elevation={1} sx={{ p:3 }}>
      <Box sx={{ display:'flex', alignItems:'center', mb:2 }}>
        <Box sx={{ color:'primary.main', mr:1 }}>{icon}</Box>
        <Typography variant="h6" fontWeight={600}>{title}</Typography>
      </Box>
      <Paper
        variant="outlined"
        sx={{ p:2, backgroundColor:'grey.50', minHeight:100, whiteSpace:'pre-wrap' }}
      >
        <Typography variant="body1">{content}</Typography>
      </Paper>
    </Paper>
  );

  return (
    <Box id="result-to-pdf" sx={{ maxWidth:1200, mx:'auto', mt:4 }}>
      
      <Card elevation={3}>
        {/* Header */}
        <Box sx={{ background:'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)', color:'white', p:3 }}>
          <Box sx={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
            <Box>
              <Typography variant="h4" fontWeight={700}>Analysis Report</Typography>
              <Typography variant="body1" sx={{ opacity:0.9 }}>
                Comprehensive service video evaluation
              </Typography>
            </Box>
            <Button
              onClick={onBack}
              startIcon={<ArrowBack />}
              variant="outlined"
              sx={{
                color:'white',
                borderColor:'white',
                '&:hover': {
                  backgroundColor:'rgba(255,255,255,0.1)',
                  borderColor:'white'
                }
              }}
            >
              Back to Dashboard
            </Button>
          </Box>
        </Box>

        {/* Body */}
        <CardContent sx={{ p:4 }}>
          {/* CitNow Info */}
          <Box sx={{ mb:6 }}>
            <Typography
              variant="h5"
              fontWeight={600}
              gutterBottom
              sx={{ display:'flex', alignItems:'center' }}
            >
              <DirectionsCar sx={{ mr:1, color:'primary.main' }}/>
              CitNow Service Information
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <InfoCard label="Dealership"      value={metadata.dealership       || 'N/A'} icon={<Business />} />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard
                  label="Vehicle"
                  value={metadata.vehicle || metadata.registration || 'N/A'}
                  icon={<DirectionsCar />}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard label="Service Advisor" value={metadata.service_advisor  || 'N/A'} icon={<Person />} />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard label="VIN"             value={metadata.vin          || 'N/A'} icon={<Description />} />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard label="Email"           value={metadata.email           || 'N/A'} icon={<Email />} />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard label="Phone"           value={metadata.phone           || 'N/A'} icon={<Phone />} />
              </Grid>
            </Grid>
          </Box>

          {/* Quality Assessment */}
          <Box sx={{ mb:6 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom>
              Quality Assessment
            </Typography>
            
            {/* Video Quality Section */}
            <Box sx={{ mb: 4 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <QualityCard
                    title="Video Quality"
                    score={videoAnalysis.quality_score}
                    label={videoAnalysis.quality_label}
                    subtitle={videoAnalysis.resolution_quality}
                    icon={<Videocam />}
                    issues={videoAnalysis.issues || []}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <QualityCard
                    title="Audio Quality"
                    score={
                      audioAnalysis.confidence != null
                        ? Math.round(audioAnalysis.confidence * 100)
                        : audioAnalysis.score || 0
                    }
                    label={audioAnalysis.prediction}
                    subtitle={audioAnalysis.clarity_level}
                    icon={<Mic />}
                    issues={audioAnalysis.issues || []}
                  />
                </Grid>
              </Grid>
            </Box>

            {/* Camera Shake Analysis */}
            {videoAnalysis.shake_level && (
              <ShakeIndicator 
                shakeLevel={videoAnalysis.shake_level}
                score={videoAnalysis.detailed_analysis?.stability?.replace('%', '')}
              />
            )}

            {/* Detailed Audio Analysis */}
            {(audioAnalysis.issues || audioAnalysis.clarity_level) && (
              <AudioQualityDetails audioAnalysis={audioAnalysis} />
            )}

            {/* Video Detailed Metrics */}
            {videoAnalysis.detailed_analysis && (
              <Paper elevation={1} sx={{ p: 2, mt: 2 }}>
                <Typography variant="h6" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <BlurOn sx={{ mr: 1, color: 'primary.main' }} />
                  Video Technical Metrics
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(videoAnalysis.detailed_analysis).map(([key, value]) => (
                    <Grid item xs={6} sm={4} md={2.4} key={key}>
                      <Box sx={{ textAlign: 'center', p: 1 }}>
                        <Typography variant="h6" color="primary.main" fontWeight={600}>
                          {typeof value === 'string' ? value.replace('%', '') : value}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                          {key.replace('_', ' ')}
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            )}
            {result.overall_quality && (
  <Paper elevation={2} sx={{ 
    p: 3, 
    mt: 3,
    background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
    border: '2px solid #0ea5e9'
  }}>
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
      <Score sx={{ color: '#0ea5e9', mr: 2, fontSize: 32 }} />
      <Box>
        <Typography variant="h5" fontWeight={700} color="#0c4a6e">
          Overall Quality Score
        </Typography>
        <Typography variant="body2" color="#0369a1">
          Combined assessment of audio and video quality
        </Typography>
      </Box>
    </Box>
    
    <Grid container spacing={3} alignItems="center">
      {/* Overall Score Circle */}
      <Grid item xs={12} md={4}>
        <Box sx={{ 
          position: 'relative', 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Box sx={{ position: 'relative', width: 120, height: 120 }}>
            <CircularProgress
              variant="determinate"
              value={100}
              size={120}
              thickness={4}
              sx={{ color: '#e0f2fe' }}
            />
            <CircularProgress
              variant="determinate"
              // ✅ FIX: Multiply by 10 to convert 0-10 scale to 0-100 for CircularProgress
              value={(result.overall_quality.overall_score || 0) * 10}
              size={120}
              thickness={4}
              sx={{ 
                color: result.overall_quality.overall_score >= 8 ? '#10b981' : 
                      result.overall_quality.overall_score >= 6 ? '#f59e0b' : '#ef4444',
                position: 'absolute',
                left: 0,
                top: 0
              }}
            />
            <Box sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center'
            }}>
              <Typography variant="h4" fontWeight={700}>
                {result.overall_quality.overall_score || 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                /10
              </Typography>
            </Box>
          </Box>
          <Chip
            label={result.overall_quality.overall_label || 'Unknown'}
            color={
              result.overall_quality.overall_label === 'Excellent' ? 'success' :
              result.overall_quality.overall_label === 'Very Good' ? 'success' :
              result.overall_quality.overall_label === 'Good' ? 'primary' :
              result.overall_quality.overall_label === 'Fair' ? 'warning' : 'error'
            }
            sx={{ mt: 2, fontWeight: 600 }}
          />
        </Box>
      </Grid>

      {/* Breakdown */}
      <Grid item xs={12} md={8}>
        <Typography variant="h6" fontWeight={600} gutterBottom>
          Quality Breakdown
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
              <Videocam sx={{ color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" color="primary.main" fontWeight={600}>
                {videoAnalysis.quality_score || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Video Quality
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {/* ✅ FIX: Show video_quality instead of video_contribution if that's what you have */}
                {result.overall_quality.breakdown?.video_quality || 0}/10
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
              <Mic sx={{ color: 'secondary.main', mb: 1 }} />
              <Typography variant="h6" color="secondary.main" fontWeight={600}>
                {audioAnalysis.score || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Audio Quality
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {/* ✅ FIX: Show audio_quality instead of audio_contribution */}
                {result.overall_quality.breakdown?.audio_quality || 0}/10
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {/* Contribution breakdown */}
        <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Contribution to Overall Score:
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Box sx={{ 
              width: `${(result.overall_quality.video_contribution || 0) * 10}%`, 
              backgroundColor: '#0ea5e9', 
              height: 8, 
              borderRadius: 4 
            }} />
            <Typography variant="caption" sx={{ ml: 1 }}>
              Video: {((result.overall_quality.video_contribution || 0) * 10).toFixed(0)}%
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ 
              width: `${(result.overall_quality.audio_contribution || 0) * 10}%`, 
              backgroundColor: '#8b5cf6', 
              height: 8, 
              borderRadius: 4 
            }} />
            <Typography variant="caption" sx={{ ml: 1 }}>
              Audio: {((result.overall_quality.audio_contribution || 0) * 10).toFixed(0)}%
            </Typography>
          </Box>
        </Box>
      </Grid>
    </Grid>
  </Paper>
)}
          </Box>

          {/* Transcription / Summary / Translation */}
          <Box sx={{ '& > *': { mb:3 } }}>
            <ContentSection
              title="Transcription"
              content={transcription.text || 'N/A'}
              icon={<Description />}
            />
            <ContentSection
              title="Summary"
              content={summarization.summary || 'N/A'}
              icon={<Summarize />}
            />
            <ContentSection
              title={`Translation (${(result.target_language||'').toUpperCase()})`}
              content={translation.translated_text || 'N/A'}
              icon={<Translate />}
            />
          </Box>

          {/* PDF */}
          <Divider sx={{ my:4 }}/>
          <Box sx={{ display:'flex', justifyContent:'center' }}>
            <Button
              onClick={() => generatePDF("result-to-pdf", `citnow_${result._id}.pdf`)}
              startIcon={<PictureAsPdf />}
              variant="contained"
              size="large"
              sx={{
                py:1.5,
                px:4,
                background:'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
                '&:hover': { background:'linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%)' }
              }}
            >
              Download PDF Report
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
