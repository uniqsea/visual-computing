import { createGlobalStyle } from 'styled-components';

export const theme = {
  colors: {
    background: '#09090b', // Deep Black
    surface: '#18181b',    // Dark Gray
    surfaceHighlight: '#27272a', // Lighter Gray for backgrounds
    border: '#3f3f46',     // Zinc 700 - Slightly lighter border for visibility
    text: '#fafafa',       // White
    textSecondary: '#a1a1aa', // Gray 400
    accent: '#d9f99d',     // Neon Lime
    accentGradient: 'linear-gradient(135deg, #d9f99d 0%, #f472b6 100%)', // Lime to Pink
    success: '#4ade80',    // Green 400
    error: '#f87171',      // Red 400
  },
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.3)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.5), 0 2px 4px -2px rgb(0 0 0 / 0.5)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.5), 0 4px 6px -4px rgb(0 0 0 / 0.5)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.5), 0 8px 10px -6px rgb(0 0 0 / 0.5)',
  },
  radii: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
  }
};

export const GlobalStyle = createGlobalStyle`
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    background-color: ${(props) => props.theme.colors.background};
    color: ${(props) => props.theme.colors.text};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  h1, h2, h3, h4, h5, h6 {
    color: ${(props) => props.theme.colors.text};
    font-weight: 600;
    line-height: 1.25;
  }

  button, input, select, textarea {
    font-family: inherit;
  }

  /* Custom Scrollbar */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${(props) => props.theme.colors.background};
  }

  ::-webkit-scrollbar-thumb {
    background: ${(props) => props.theme.colors.surfaceHighlight};
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: ${(props) => props.theme.colors.textSecondary};
  }
`;
