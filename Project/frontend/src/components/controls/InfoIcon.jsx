import styled from 'styled-components';
import { Info } from 'lucide-react';

const TooltipText = styled.span`
  visibility: hidden;
  width: 200px;
  background-color: ${(props) => props.theme.colors.surfaceHighlight};
  color: ${(props) => props.theme.colors.text};
  text-align: center;
  border-radius: 6px;
  padding: 8px;
  position: absolute;
  z-index: 10;
  bottom: 125%;
  left: 50%;
  margin-left: -100px;
  opacity: 0;
  transition: opacity 0.3s;
  font-size: 0.75rem;
  font-weight: 400;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
  border: 1px solid ${(props) => props.theme.colors.border};
  pointer-events: none;

  &::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: ${(props) => props.theme.colors.surfaceHighlight} transparent transparent transparent;
  }
`;

const InfoIconWrapper = styled.span`
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: ${(props) => props.theme.colors.textSecondary};
  flex-shrink: 0;
  cursor: pointer;
  margin-left: 2px;
  transform: translateY(-2px);

  &:hover {
    color: ${(props) => props.theme.colors.accent};
  }

  &:hover ${TooltipText} {
    visibility: visible;
    opacity: 1;
  }
`;

export default function InfoIcon({ title }) {
    return (
        <InfoIconWrapper>
            <Info size={12} />
            <TooltipText>{title}</TooltipText>
        </InfoIconWrapper>
    );
}
