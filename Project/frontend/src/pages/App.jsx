import { useState } from 'react';
import styled, { ThemeProvider } from 'styled-components';
import { GlobalStyle, theme } from '../styles/theme';
import ControlsPanel from '../components/ControlsPanel';
import RenderPreview from '../components/RenderPreview';

const AppWrapper = styled.div`
  display: flex;
  height: 100vh;
  background: ${(props) => props.theme.colors.background};
  color: ${(props) => props.theme.colors.text};
  overflow: hidden;
`;

const MainLayout = styled.div`
  display: grid;
  grid-template-columns: minmax(300px, 320px) 1fr;
  flex-grow: 1;
  overflow: hidden;
`;

const Sidebar = styled.div`
  padding: 0;
  border-right: 1px solid ${(props) => props.theme.colors.border};
  overflow-y: hidden;
  background: ${(props) => props.theme.colors.background};
  display: flex;
  flex-direction: column;
`;

const Content = styled.main`
  padding: 24px;
  overflow-y: auto;
  background: ${(props) => props.theme.colors.background};
  display: flex;
  flex-direction: column;
`;

function App() {
  const [mode, setMode] = useState('heightmap');
  const [evaluationMode, setEvaluationMode] = useState(false);
  const [evaluationFiles, setEvaluationFiles] = useState([]);
  const [selectedEvalIndex, setSelectedEvalIndex] = useState(0);
  const [settingsSnapshot, setSettingsSnapshot] = useState({
    extrusion: {},
    revolution: {},
    heightmap: {}
  });

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <AppWrapper>
        <MainLayout>
          <Sidebar>
            <ControlsPanel
              mode={mode}
              setMode={setMode}
              evaluationMode={evaluationMode}
              setEvaluationMode={setEvaluationMode}
              evaluationFiles={evaluationFiles}
              setEvaluationFiles={setEvaluationFiles}
              selectedEvalIndex={selectedEvalIndex}
              setSelectedEvalIndex={setSelectedEvalIndex}
              onSettingsChange={setSettingsSnapshot}
            />
          </Sidebar>
          <Content>
            <RenderPreview
              mode={mode}
              evaluationMode={evaluationMode}
              evaluationFiles={evaluationFiles}
              selectedEvalIndex={selectedEvalIndex}
              setSelectedEvalIndex={setSelectedEvalIndex}
            />
          </Content>
        </MainLayout>
      </AppWrapper>
    </ThemeProvider>
  );
}

export default App;
