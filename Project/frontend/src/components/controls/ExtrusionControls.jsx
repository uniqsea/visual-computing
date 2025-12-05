import styled from 'styled-components';
import SliderControl from './SliderControl';

const FieldGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 12px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: ${(props) => props.theme.radii.md};
`;

export default function ExtrusionControls({
    depth, setDepth,
    smoothSteps, setSmoothSteps,
    thickness, setThickness
}) {
    return (
        <FieldGroup>
            <SliderControl
                label="Extrusion Depth"
                description="Distance the sketch is extruded along the Z axis."
                value={depth}
                onChange={setDepth}
                min={0.05}
                max={1}
                step={0.05}
                formatValue={(v) => v.toFixed(2)}
            />
            <SliderControl
                label="Smooth Steps"
                description="Number of contour subdivision steps to smooth jagged lines."
                value={smoothSteps}
                onChange={setSmoothSteps}
                min={0}
                max={3}
                step={1}
            />
            <SliderControl
                label="Stroke Thickness"
                description="Additional line thickness applied before extraction; larger values produce fatter shapes."
                value={thickness}
                onChange={setThickness}
                min={0}
                max={5}
                step={0.5}
                formatValue={(v) => v.toFixed(1)}
            />
        </FieldGroup>
    );
}
