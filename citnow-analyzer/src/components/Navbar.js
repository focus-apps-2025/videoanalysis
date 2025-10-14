import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import { Analytics, History, Upload } from '@mui/icons-material';


export default function Navbar({ activeTab, onTabChange, onBack }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <AppBar 
      position="static" 
      elevation={0}
      sx={{ 
        backgroundColor: 'white',
        borderBottom: '1px solid',
        borderColor: 'grey.100',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between', py: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box
            sx={{
              width: 32,
              height: 32,
              backgroundColor: 'primary.main',
              borderRadius: 2,
              mr: 2,
            }}
          />
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              color: 'transparent',
            }}
          >
            CitNow Analyzer
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            startIcon={<Analytics />}
            onClick={() => onTabChange('new')}
            variant={activeTab === 'new' ? 'contained' : 'text'}
            size={isMobile ? 'small' : 'medium'}
            sx={{
              borderRadius: 3,
              ...(activeTab === 'new' && {
                backgroundColor: 'primary.main',
              }),
            }}
          >
            {isMobile ? 'New' : 'New Analysis'}
          </Button>
          
          <Button
            startIcon={<Upload />}
            onClick={() => onTabChange('bulk')}
            variant={activeTab === 'bulk' ? 'contained' : 'text'}
            size={isMobile ? 'small' : 'medium'}
            sx={{
              borderRadius: 3,
              ...(activeTab === 'bulk' && {
                backgroundColor: 'primary.main',
              }),
            }}
          >
            {isMobile ? 'Bulk' : 'Bulk Upload'}
          </Button>
          
          <Button
            startIcon={<History />}
            onClick={() => onTabChange('history')}
            variant={activeTab === 'history' ? 'contained' : 'text'}
            size={isMobile ? 'small' : 'medium'}
            sx={{
              borderRadius: 3,
              ...(activeTab === 'history' && {
                backgroundColor: 'primary.main',
              }),
            }}
          >
            {isMobile ? 'History' : 'View History'}
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}
