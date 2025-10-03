import React from 'react';
import {
  Box, Card, CardContent, Typography, Button, Grid,
  Paper, Chip, Divider
} from '@mui/material';
import {
  ArrowBack, PictureAsPdf, DirectionsCar, Person,
  Phone, Email, Videocam, Mic, Description, Translate,
  Business, Summarize
} from '@mui/icons-material';
import { generatePDF } from './pdfgenerator';

export default function AnalysisDetails({ result: rawResult, onBack }) {
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

  // 3) Re-usable UI pieces
  const InfoCard = ({ label, value, icon }) => (
    <Paper elevation={1} sx={{ p:2, height:'100%' }}>
      <Box sx={{ display:'flex', alignItems:'center', mb:1 }}>
        <Box sx={{ color:'primary.main', mr:1 }}>{icon}</Box>
        <Typography variant="subtitle2" color="text.secondary">{label}</Typography>
      </Box>
      <Typography variant="body1" fontWeight={500}>{value}</Typography>
    </Paper>
  );

  const QualityCard = ({ title, score, label, icon }) => (
    <Paper elevation={2} sx={{
      p:3,
      background:'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
      border:'1px solid grey.100'
    }}>
      <Box sx={{ display:'flex', alignItems:'center', mb:2 }}>
        <Box sx={{ p:1, backgroundColor:'primary.light', borderRadius:2, mr:2, color:'white' }}>
          {icon}
        </Box>
        <Typography variant="h6" fontWeight={600}>{title}</Typography>
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
      />
    </Paper>
  );

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
                <InfoCard label="VIN"             value={metadata.vin             || 'N/A'} icon={<Description />} />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard label="Email"           value={metadata.email           || 'N/A'} icon={<Email />} />
              </Grid>
              <Grid item xs={12} md={6}>
                <InfoCard label="Phone"           value={metadata.phone           || 'N/A'} icon={<Phone />} />
              </Grid>
            </Grid>
          </Box>

          {/* Quality */}
          <Box sx={{ mb:6 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom>
              Quality Assessment
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <QualityCard
                  title="Video Quality"
                  score={videoAnalysis.quality_score}
                  label={videoAnalysis.quality_label}
                  note={videoAnalysis.quality_note}
                  icon={<Videocam />}
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
                  note={videoAnalysis.quality_note}
                  icon={<Mic />}
                />
              </Grid>
            </Grid>
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
