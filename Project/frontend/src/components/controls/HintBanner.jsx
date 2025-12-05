import styled from 'styled-components';
import { Lightbulb, X } from 'lucide-react';

const BannerContainer = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 16px;
  background: rgba(39, 39, 42, 0.9);
  backdrop-filter: blur(8px);
  border-left: 3px solid ${(props) => props.theme.colors.accent};
  border-radius: 4px;
  margin-bottom: 0;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
  position: relative;
`;

const IconWrapper = styled.div`
  color: ${(props) => props.theme.colors.accent};
  display: flex;
  align-items: center;
  margin-top: 2px;
`;

const TextContent = styled.div`
  color: ${(props) => props.theme.colors.textSecondary};
  font-size: 0.85rem;
  line-height: 1.5;
  flex: 1;
  padding-right: 16px;
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: ${(props) => props.theme.colors.textSecondary};
  cursor: pointer;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0.7;
  transition: opacity 0.2s;

  &:hover {
    opacity: 1;
    color: ${(props) => props.theme.colors.text};
  }
`;

export default function HintBanner({ children, onDismiss }) {
  return (
    <BannerContainer>
      <IconWrapper>
        <Lightbulb size={18} />
      </IconWrapper>
      <TextContent>{children}</TextContent>
      {onDismiss && (
        <CloseButton onClick={onDismiss}>
          <X size={14} />
        </CloseButton>
      )}
    </BannerContainer>
  );
}
