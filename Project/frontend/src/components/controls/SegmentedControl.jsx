import styled from 'styled-components';

const Container = styled.div`
  display: flex;
  background: ${(props) => props.theme.colors.surfaceHighlight};
  padding: 4px;
  border-radius: ${(props) => props.theme.radii.lg};
  gap: 4px;
`;

const Option = styled.button`
  flex: 1;
  padding: 6px 2px;
  border: none;
  background: ${(props) => (props.$active ? props.theme.colors.accent : 'transparent')};
  color: ${(props) => (props.$active ? '#000000' : props.theme.colors.textSecondary)};
  font-weight: 600;
  font-size: 0.8rem;
  border-radius: ${(props) => props.theme.radii.md};
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  
  &:hover {
    color: ${(props) => (props.$active ? '#000000' : props.theme.colors.text)};
    background: ${(props) => (props.$active ? props.theme.colors.accent : 'rgba(255, 255, 255, 0.05)')};
  }
`;

export default function SegmentedControl({ options, value, onChange }) {
    return (
        <Container>
            {options.map((option) => (
                <Option
                    key={option.value}
                    $active={value === option.value}
                    onClick={() => onChange(option.value)}
                >
                    {option.label}
                </Option>
            ))}
        </Container>
    );
}
