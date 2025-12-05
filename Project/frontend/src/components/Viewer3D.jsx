import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useEffect, useMemo, useState } from 'react';
import { BufferGeometry, Float32BufferAttribute, DoubleSide } from 'three';
import { theme } from '../styles/theme';

function MeshContent({ meshUrl }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    let mounted = true;
    fetch(meshUrl)
      .then((response) => response.json())
      .then((json) => {
        if (mounted) {
          setData(json);
        }
      })
      .catch(() => setData(null));
    return () => {
      mounted = false;
    };
  }, [meshUrl]);

  const geometry = useMemo(() => {
    if (!data) return null;
    const geo = new BufferGeometry();
    geo.setAttribute('position', new Float32BufferAttribute(new Float32Array(data.positions ?? []), 3));
    if (data.normals) {
      geo.setAttribute('normal', new Float32BufferAttribute(new Float32Array(data.normals), 3));
    }
    if (data.indices) {
      geo.setIndex(data.indices);
    } else {
      geo.computeVertexNormals();
    }
    geo.center();
    geo.computeBoundingSphere();
    return geo;
  }, [data]);

  if (!geometry) {
    return null;
  }

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={theme.colors.accent} flatShading side={DoubleSide} />
    </mesh>
  );
}

function Viewer3D({ meshUrl, cameraPosition = [0, 0, 2], target = [0, 0, 0] }) {
  return (
    <Canvas
      style={{ height: '100%', width: '100%', borderRadius: 8 }}
      camera={{ position: cameraPosition, fov: 50 }}
    >
      <color attach="background" args={[theme.colors.surface]} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[3, 4, 5]} intensity={1.5} />
      <gridHelper args={[10, 20, '#444', '#333']} />
      <MeshContent meshUrl={meshUrl} />
      <OrbitControls target={target} />
    </Canvas>
  );
}

export default Viewer3D;
