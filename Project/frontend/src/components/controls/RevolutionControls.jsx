import styled from 'styled-components';
import SliderControl from './SliderControl';
import CheckboxControl from './CheckboxControl';

const FieldGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 12px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: ${(props) => props.theme.radii.md};
`;

export default function RevolutionControls({
    segments, setSegments,
    axisOffsetX, setAxisOffsetX,
    angle, setAngle,
    capBottom, setCapBottom,
    capTop, setCapTop,
    hollow, setHollow,
    wallThickness, setWallThickness
}) {
    return (
        <FieldGroup>
            <SliderControl
                label="Segments"
                description="Number of slices around the rotation; higher values look smoother."
                value={segments}
                onChange={setSegments}
                min={16}
                max={128}
                step={4}
            />
            <SliderControl
                label="Axis Offset X"
                description="Horizontal position of the revolution axis relative to the sketch."
                value={axisOffsetX}
                onChange={setAxisOffsetX}
                min={-0.5}
                max={0.5}
                step={0.01}
                formatValue={(v) => v.toFixed(2)}
            />
            <SliderControl
                label="Sweep Angle"
                description="Total angle of rotation; less than 360° leaves an open arc."
                value={angle}
                onChange={setAngle}
                min={10}
                max={360}
                step={10}
                formatValue={(v) => `${v}°`}
            />

            <CheckboxControl
                label="Cap Bottom"
                description="Close the bottom face of the revolution."
                checked={capBottom}
                onChange={setCapBottom}
            />

            <CheckboxControl
                label="Cap Top"
                description="Close the top face of the revolution."
                checked={capTop}
                onChange={setCapTop}
            />

            <CheckboxControl
                label="Hollow"
                description="Make the shape a shell; adjust Wall Thickness to control inner radius."
                checked={hollow}
                onChange={setHollow}
            />

            {hollow && (
                <SliderControl
                    label="Wall Thickness"
                    description="Thickness of the hollow shell."
                    value={wallThickness}
                    onChange={setWallThickness}
                    min={0.01}
                    max={0.2}
                    step={0.01}
                    formatValue={(v) => v.toFixed(2)}
                />
            )}
        </FieldGroup>
    );
}
