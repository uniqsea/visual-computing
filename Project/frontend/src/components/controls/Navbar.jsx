import styled from 'styled-components';
import SegmentedControl from './SegmentedControl';

const NavbarContainer = styled.div`
  padding: 16px 24px;
  border-bottom: 1px solid ${(props) => props.theme.colors.border};
  background: ${(props) => props.theme.colors.surface};
  flex-shrink: 0;
`;

export default function Navbar({ evaluationMode, setEvaluationMode }) {
    return (
        <NavbarContainer>
            <SegmentedControl
                value={evaluationMode ? 'evaluation' : 'interactive'}
                onChange={(val) => setEvaluationMode(val === 'evaluation')}
                options={[
                    { label: 'Interactive', value: 'interactive' },
                    { label: 'Evaluation', value: 'evaluation' }
                ]}
            />
        </NavbarContainer>
    );
}
