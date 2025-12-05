import styled from 'styled-components';
import InfoIcon from './InfoIcon';

const ControlRow = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const LabelRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
`;

const FieldLabel = styled.span`
  font-size: 0.9rem;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textSecondary};
  display: inline-flex;
  align-items: center;
  gap: 4px;
`;

const ValueLabel = styled.span`
  font-size: 0.75rem;
  font-family: monospace;
  color: ${(props) => props.theme.colors.accent};
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
`;

const RangeInput = styled.input`
  -webkit-appearance: none;
  width: 100%;
  background: transparent;
  cursor: pointer;
  height: 20px;

  &:focus {
    outline: none;
  }

  &::-webkit-slider-runnable-track {
    width: 100%;
    height: 4px;
    cursor: pointer;
    background: ${(props) => props.theme.colors.surfaceHighlight};
    border-radius: 2px;
  }

  &::-webkit-slider-thumb {
    height: 14px;
    width: 14px;
    border-radius: 50%;
    background: ${(props) => props.theme.colors.accent};
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -5px;
    transition: transform 0.1s;
    box-shadow: 0 0 0 2px ${(props) => props.theme.colors.surface};
    
    &:hover {
      transform: scale(1.2);
    }
  }
`;

export default function SliderControl({
    label,
    description,
    value,
    onChange,
    min,
    max,
    step,
    formatValue = (v) => v
}) {
    return (
        <ControlRow>
            <LabelRow>
                <FieldLabel>
                    {label}
                    {description && <InfoIcon title={description} />}
                </FieldLabel>
                <ValueLabel>{formatValue(value)}</ValueLabel>
            </LabelRow>
            <RangeInput
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
            />
        </ControlRow>
    );
}
