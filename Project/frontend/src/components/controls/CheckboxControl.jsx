import styled from 'styled-components';
import InfoIcon from './InfoIcon';

const CheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  font-size: 0.8rem;
  color: ${(props) => props.theme.colors.textSecondary};
  user-select: none;
  padding: 4px 0;
  
  &:hover {
    color: ${(props) => props.theme.colors.text};
  }
`;

const CheckboxInput = styled.input`
  appearance: none;
  width: 16px;
  height: 16px;
  border: 2px solid ${(props) => props.theme.colors.border};
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
  position: relative;
  transition: all 0.2s;
  margin: 0;

  &:checked {
    background: ${(props) => props.theme.colors.accent};
    border-color: ${(props) => props.theme.colors.accent};
  }

  &:checked::after {
    content: '✓';
    position: absolute;
    color: black;
    font-size: 10px;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-weight: 800;
  }
  
  &:hover {
    border-color: ${(props) => props.theme.colors.textSecondary};
  }
`;

export default function CheckboxControl({ label, description, checked, onChange }) {
    return (
        <CheckboxLabel>
            <CheckboxInput
                type="checkbox"
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
            />
            {label}
            {description && <InfoIcon title={description} />}
        </CheckboxLabel>
    );
}
