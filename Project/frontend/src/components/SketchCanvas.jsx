import { useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { Brush, Square, Circle, Minus, Save, Trash2, RotateCcw } from 'lucide-react';

const CANVAS_WIDTH = 576;
const CANVAS_HEIGHT = 640;
const MAX_HISTORY = 30;
const ELLIPSE_SEGMENTS = 36;

const Wrapper = styled.div`
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: ${(props) => props.theme.radii.lg};
  overflow: hidden;
  position: relative;
  background: #ffffff;
  width: 100%;
  flex-shrink: 0;
`;

const SvgSurface = styled.svg`
  width: 100%;
  height: 320px;
  display: block;
  cursor: crosshair;
  background: #ffffff;
  touch-action: none;
`;

const AxisLine = styled.line`
  stroke: rgba(0, 0, 0, 0.35);
  stroke-width: 1;
  pointer-events: none;
`;

const Toolbar = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 12px;
`;

const ToolButton = styled.button`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px;
  border-radius: ${(props) => props.theme.radii.md};
  border: 1px solid
    ${(props) => (props.$active ? props.theme.colors.accent : props.theme.colors.border)};
  background: ${(props) =>
    props.$active ? props.theme.colors.surfaceHighlight : props.theme.colors.surface};
  color: ${(props) =>
    props.$active ? props.theme.colors.accent : props.theme.colors.text};
  cursor: pointer;
  transition: all 0.2s ease;
`;

const Message = styled.small`
  display: block;
  margin-top: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
  font-size: 0.85rem;
`;

const BrushControls = styled.div`
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const BrushLabel = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
  color: ${(props) => props.theme.colors.textSecondary};
  font-weight: 600;
`;

const BrushValue = styled.span`
  font-family: monospace;
  color: ${(props) => props.theme.colors.accent};
`;

const Range = styled.input`
  width: 100%;
  -webkit-appearance: none;
  height: 18px;
  background: transparent;

  &::-webkit-slider-runnable-track {
    width: 100%;
    height: 4px;
    background: ${(props) => props.theme.colors.surfaceHighlight};
    border-radius: 2px;
  }

  &::-webkit-slider-thumb {
    height: 14px;
    width: 14px;
    border-radius: 50%;
    background: ${(props) => props.theme.colors.accent};
    -webkit-appearance: none;
    margin-top: -5px;
    box-shadow: 0 0 0 2px ${(props) => props.theme.colors.surface};
  }
`;

const TOOL_CONFIG = [
  { id: 'brush', icon: Brush, label: 'Brush' },
  { id: 'rect', icon: Square, label: 'Rectangle' },
  { id: 'ellipse', icon: Circle, label: 'Ellipse' },
  { id: 'line', icon: Minus, label: 'Line' }
];

const createId = (() => {
  let counter = 0;
  return () => {
    counter += 1;
    return `shape-${Date.now()}-${counter}`;
  };
})();

function buildPathD(points, closed) {
  if (!points.length) {
    return '';
  }
  const cmds = points.map((pt) => `${pt.x} ${pt.y}`);
  const body =
    points.length === 1
      ? `M ${cmds[0]} L ${cmds[0]}`
      : `M ${cmds[0]} L ${cmds.slice(1).join(' ')}`;
  return closed ? `${body} Z` : body;
}

function cloneShapes(shapes) {
  return shapes.map((shape) => ({
    ...shape,
    points: shape.points.map((pt) => ({ ...pt }))
  }));
}

function approximateEllipse(start, current) {
  const minX = Math.min(start.x, current.x);
  const minY = Math.min(start.y, current.y);
  const width = Math.abs(current.x - start.x);
  const height = Math.abs(current.y - start.y);
  const cx = minX + width / 2;
  const cy = minY + height / 2;
  const rx = width / 2;
  const ry = height / 2;
  const pts = [];
  for (let i = 0; i < ELLIPSE_SEGMENTS; i += 1) {
    const theta = (i / ELLIPSE_SEGMENTS) * Math.PI * 2;
    pts.push({ x: cx + rx * Math.cos(theta), y: cy + ry * Math.sin(theta) });
  }
  return pts;
}

function SketchCanvas({
  onSketchSaved,
  showRevolutionAxis = false,
  axisOffsetX = 0
}) {
  const svgRef = useRef(null);
  const drawingRef = useRef(false);
  const [tool, setTool] = useState('brush');
  const [brushSize, setBrushSize] = useState(4);
  const [shapes, setShapes] = useState([]);
  const [draftShape, setDraftShape] = useState(null);
  const [message, setMessage] = useState('');
  const historyRef = useRef([]);
  const axisX = useMemo(() => {
    const clamped = Math.max(-0.5, Math.min(0.5, axisOffsetX || 0));
    return (0.5 + clamped) * CANVAS_WIDTH;
  }, [axisOffsetX]);

  const pushHistory = (snapshot) => {
    historyRef.current.push(snapshot);
    if (historyRef.current.length > MAX_HISTORY) {
      historyRef.current.shift();
    }
  };

  useEffect(() => {
    pushHistory(cloneShapes([]));
  }, []);

  const getSvgPoint = (event) => {
    if (!svgRef.current) return { x: 0, y: 0 };
    const rect = svgRef.current.getBoundingClientRect();
    const clientX = event.clientX ?? event.touches?.[0]?.clientX ?? 0;
    const clientY = event.clientY ?? event.touches?.[0]?.clientY ?? 0;
    const x = ((clientX - rect.left) / rect.width) * CANVAS_WIDTH;
    const y = ((clientY - rect.top) / rect.height) * CANVAS_HEIGHT;
    return {
      x: Math.max(0, Math.min(CANVAS_WIDTH, x)),
      y: Math.max(0, Math.min(CANVAS_HEIGHT, y))
    };
  };

  const finalizeShape = (shape) => {
    if (!shape || !shape.points || shape.points.length < 2) {
      return null;
    }
    return {
      id: createId(),
      points: shape.points.map((pt) => ({ ...pt })),
      closed: shape.closed,
      strokeWidth: shape.strokeWidth
    };
  };

  const addShape = (shape) => {
    if (!shape) return;
    setShapes((prev) => {
      const next = [...prev, shape];
      pushHistory(cloneShapes(next));
      return next;
    });
  };

  const handlePointerDown = (event) => {
    event.preventDefault();
    drawingRef.current = true;
    const point = getSvgPoint(event);
    if (tool === 'brush') {
      setDraftShape({
        tool,
        points: [point],
        strokeWidth: brushSize,
        closed: false
      });
    } else {
      setDraftShape({
        tool,
        start: point,
        current: point,
        strokeWidth: brushSize
      });
    }
  };

  const handlePointerMove = (event) => {
    if (!drawingRef.current) return;
    event.preventDefault();
    const point = getSvgPoint(event);
    setDraftShape((prev) => {
      if (!prev) return prev;
      if (prev.tool === 'brush') {
        return {
          ...prev,
          points: [...prev.points, point]
        };
      }
      return { ...prev, current: point };
    });
  };

  const handlePointerUp = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    setDraftShape((prev) => {
      if (!prev) return null;
      let points = [];
      let closed = false;
      if (prev.tool === 'brush') {
        points = prev.points;
        closed = false;
      } else if (prev.tool === 'line') {
        points = [prev.start, prev.current];
        closed = false;
      } else if (prev.tool === 'rect') {
        const start = prev.start;
        const current = prev.current;
        points = [
          { x: start.x, y: start.y },
          { x: current.x, y: start.y },
          { x: current.x, y: current.y },
          { x: start.x, y: current.y }
        ];
        closed = true;
      } else if (prev.tool === 'ellipse') {
        points = approximateEllipse(prev.start, prev.current);
        closed = true;
      }
      const normalized = finalizeShape({
        points,
        closed,
        strokeWidth: prev.strokeWidth
      });
      addShape(normalized);
      return null;
    });
  };

  const undoLast = () => {
    if (historyRef.current.length <= 1) {
      setMessage('Nothing to undo yet.');
      return;
    }
    historyRef.current.pop();
    const snapshot = historyRef.current[historyRef.current.length - 1];
    setShapes(snapshot);
    setMessage('Undid last action.');
    if (onSketchSaved) {
      onSketchSaved(null);
    }
  };

  const clearCanvas = () => {
    setShapes([]);
    pushHistory(cloneShapes([]));
    setMessage('Canvas cleared.');
    if (onSketchSaved) {
      onSketchSaved(null);
    }
  };

  const exportSvg = () => {
    if (!shapes.length) return null;
    const metadata = {
      width: CANVAS_WIDTH,
      height: CANVAS_HEIGHT,
      shapes: shapes.map((shape) => ({
        points: shape.points.map((pt) => [
          Number(pt.x.toFixed(2)),
          Number(pt.y.toFixed(2))
        ]),
        closed: shape.closed,
        strokeWidth: shape.strokeWidth
      }))
    };
    const metaString = JSON.stringify(metadata)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    const pathElements = shapes
      .map((shape) => {
        const d = buildPathD(shape.points, shape.closed);
        const fill = shape.closed ? 'none' : 'none';
        return `<path d="${d}" fill="${fill}" stroke="#000" stroke-width="${shape.strokeWidth}" stroke-linecap="round" stroke-linejoin="round" />`;
      })
      .join('\n');
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${CANVAS_WIDTH}" height="${CANVAS_HEIGHT}" viewBox="0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}">
<metadata id="sketch-data">${metaString}</metadata>
<g fill="none">${pathElements}</g>
</svg>`;
  };

  const saveSketch = () => {
    const svgContent = exportSvg();
    if (!svgContent) {
      setMessage('Nothing to save yet.');
      return;
    }
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const file = new File([blob], 'sketch.svg', { type: 'image/svg+xml' });
    if (onSketchSaved) {
      onSketchSaved(file);
      setMessage('Sketch saved. You can now submit it.');
    }
  };

  const previewShapes = useMemo(() => {
    const renderShape = (shape, key) => {
      const d = buildPathD(shape.points, shape.closed);
      return (
        <path
          key={key}
          d={d}
          fill="none"
          stroke="#000000"
          strokeWidth={shape.strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      );
    };
    const items = shapes.map((shape) => renderShape(shape, shape.id));
    if (draftShape) {
      let preview = null;
      if (draftShape.tool === 'brush') {
        preview = {
          points: draftShape.points || [],
          closed: false,
          strokeWidth: draftShape.strokeWidth
        };
      } else if (draftShape.tool === 'line') {
        preview = {
          points: draftShape.current
            ? [draftShape.start, draftShape.current]
            : [draftShape.start, draftShape.start],
          closed: false,
          strokeWidth: draftShape.strokeWidth
        };
      } else if (draftShape.tool === 'rect') {
        preview = {
          points: draftShape.current
            ? [
              { x: draftShape.start.x, y: draftShape.start.y },
              { x: draftShape.current.x, y: draftShape.start.y },
              { x: draftShape.current.x, y: draftShape.current.y },
              { x: draftShape.start.x, y: draftShape.current.y }
            ]
            : [draftShape.start],
          closed: true,
          strokeWidth: draftShape.strokeWidth
        };
      } else if (draftShape.tool === 'ellipse' && draftShape.current) {
        preview = {
          points: approximateEllipse(draftShape.start, draftShape.current),
          closed: true,
          strokeWidth: draftShape.strokeWidth
        };
      }
      if (preview) {
        items.push(renderShape(preview, 'draft'));
      }
    }
    return items;
  }, [shapes, draftShape]);

  return (
    <div>
      <Wrapper>
        <SvgSurface
          ref={svgRef}
          viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
        >
          {showRevolutionAxis && (
            <AxisLine
              x1={axisX}
              y1={0}
              x2={axisX}
              y2={CANVAS_HEIGHT}
            />
          )}
          {previewShapes}
        </SvgSurface>
      </Wrapper>
      <Toolbar>
        {TOOL_CONFIG.map(({ id, icon: Icon }) => (
          <ToolButton
            key={id}
            type="button"
            $active={tool === id}
            onClick={() => setTool(id)}
          >
            <Icon size={20} />
          </ToolButton>
        ))}
      </Toolbar>
      <Toolbar>
        <ToolButton type="button" onClick={saveSketch}>
          <Save size={20} />
        </ToolButton>
        <ToolButton type="button" onClick={undoLast}>
          <RotateCcw size={20} />
        </ToolButton>
        <ToolButton type="button" onClick={clearCanvas}>
          <Trash2 size={20} />
        </ToolButton>
      </Toolbar>
      <BrushControls>
        <BrushLabel>
          Line Width <BrushValue>{brushSize.toFixed(1)} px</BrushValue>
        </BrushLabel>
        <Range
          type="range"
          min="1"
          max="20"
          step="0.5"
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value) || 4)}
        />
      </BrushControls>
      {message && <Message>{message}</Message>}
    </div>
  );
}

export default SketchCanvas;
