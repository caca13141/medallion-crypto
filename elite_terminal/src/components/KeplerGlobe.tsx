import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Html } from '@react-three/drei';

interface ClusterNode {
    id: number;
    label: string;
    size: number;
    color: string;
    phi: number; // Spherical coordinate
    theta: number; // Spherical coordinate
}

interface KeplerMapperData {
    nodes: ClusterNode[];
    edges: [number, number][];
}

export function KeplerGlobe({ data }: { data: KeplerMapperData | null }) {
    const groupRef = useRef<THREE.Group>(null!);
    const [hoveredNode, setHoveredNode] = useState<number | null>(null);

    // Generate mock KeplerMapper graph if no data
    const graphData = useMemo(() => {
        if (data) return data;

        const nodeCount = 20;
        const nodes: ClusterNode[] = [];
        const edges: [number, number][] = [];

        for (let i = 0; i < nodeCount; i++) {
            const phi = Math.acos(-1 + (2 * i) / nodeCount);
            const theta = Math.sqrt(nodeCount * Math.PI) * phi;

            nodes.push({
                id: i,
                label: `Cluster ${i}`,
                size: 0.2 + Math.random() * 0.3,
                color: i % 3 === 0 ? '#00ffff' : i % 3 === 1 ? '#ff0055' : '#ffaa00',
                phi,
                theta
            });

            // Connect to nearby nodes
            if (i > 0) edges.push([i - 1, i]);
            if (i > 3) edges.push([i - 3, i]);
        }

        return { nodes, edges };
    }, [data]);

    useFrame((state) => {
        if (groupRef.current) {
            groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.1;
        }
    });

    return (
        <group ref={groupRef}>
            {/* Globe wireframe */}
            <mesh>
                <sphereGeometry args={[2, 32, 32]} />
                <meshBasicMaterial color="#0a0a0a" wireframe opacity={0.1} transparent />
            </mesh>

            {/* Equator ring */}
            <mesh rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[2, 0.01, 16, 100]} />
                <meshBasicMaterial color="#333" />
            </mesh>

            {/* Meridian ring */}
            <mesh>
                <torusGeometry args={[2, 0.01, 16, 100]} />
                <meshBasicMaterial color="#333" />
            </mesh>

            {/* Cluster Nodes */}
            {graphData.nodes.map((node) => {
                const x = 2 * Math.sin(node.phi) * Math.cos(node.theta);
                const y = 2 * Math.sin(node.phi) * Math.sin(node.theta);
                const z = 2 * Math.cos(node.phi);

                const isHovered = hoveredNode === node.id;

                return (
                    <group key={node.id} position={[x, y, z]}>
                        <mesh
                            onPointerOver={() => setHoveredNode(node.id)}
                            onPointerOut={() => setHoveredNode(null)}
                        >
                            <sphereGeometry args={[node.size * (isHovered ? 1.5 : 1), 16, 16]} />
                            <meshStandardMaterial
                                color={node.color}
                                emissive={node.color}
                                emissiveIntensity={isHovered ? 0.8 : 0.3}
                            />
                        </mesh>
                        {isHovered && (
                            <Html distanceFactor={10}>
                                <div className="bg-black/90 text-cyan-500 px-2 py-1 rounded text-xs border border-cyan-900/50 backdrop-blur-sm pointer-events-none">
                                    {node.label}
                                </div>
                            </Html>
                        )}
                    </group>
                );
            })}

            {/* Edges between clusters */}
            {graphData.edges.map(([from, to], i) => {
                const fromNode = graphData.nodes[from];
                const toNode = graphData.nodes[to];

                if (!fromNode || !toNode) return null;

                const start = new THREE.Vector3(
                    2 * Math.sin(fromNode.phi) * Math.cos(fromNode.theta),
                    2 * Math.sin(fromNode.phi) * Math.sin(fromNode.theta),
                    2 * Math.cos(fromNode.phi)
                );

                const end = new THREE.Vector3(
                    2 * Math.sin(toNode.phi) * Math.cos(toNode.theta),
                    2 * Math.sin(toNode.phi) * Math.sin(toNode.theta),
                    2 * Math.cos(toNode.phi)
                );

                const curve = new THREE.QuadraticBezierCurve3(
                    start,
                    start.clone().lerp(end, 0.5).multiplyScalar(1.2), // Bulge out
                    end
                );

                const points = curve.getPoints(50);
                const geometry = new THREE.BufferGeometry().setFromPoints(points);

                return (
                    <line key={i} geometry={geometry}>
                        <lineBasicMaterial color="#333" opacity={0.3} transparent />
                    </line>
                );
            })}

            {/* Ambient particles */}
            <points>
                <bufferGeometry>
                    <bufferAttribute
                        attach="attributes-position"
                        count={100}
                        array={new Float32Array(
                            Array.from({ length: 300 }, () => (Math.random() - 0.5) * 5)
                        )}
                        itemSize={3}
                    />
                </bufferGeometry>
                <pointsMaterial size={0.02} color="#00ffff" transparent opacity={0.4} />
            </points>
        </group>
    );
}

// Import useState
import { useState } from 'react';
