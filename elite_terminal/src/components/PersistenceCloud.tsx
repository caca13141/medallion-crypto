import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Points, PointMaterial, Line } from '@react-three/drei';
import * as THREE from 'three';

interface TopologySnapshot {
    persistence_image: number[][];
    betti_curves: number[][];
    wasserstein_dist: number;
    // Assuming raw diagram data might be passed or simulated for now
    // [birth, death, dimension]
    diagram?: number[][];
}

export function PersistenceCloud({ data }: { data: TopologySnapshot | null }) {
    const pointsRef = useRef<THREE.Points>(null!);
    const linesRef = useRef<THREE.Group>(null!);

    // Generate mock diagram data if none exists (for visualization)
    const particles = useMemo(() => {
        const count = 200;
        const pts = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const sizes = new Float32Array(count);
        const trails = [];

        for (let i = 0; i < count; i++) {
            // Birth (x), Death (y), Persistence (z/size)
            const birth = (Math.random() - 0.5) * 3;
            const death = birth + Math.random() * 2; // Death always > Birth
            const persistence = death - birth;

            pts[i * 3] = birth;
            pts[i * 3 + 1] = death;
            pts[i * 3 + 2] = persistence;

            // Color based on persistence (Red = High, Cyan = Low)
            const color = new THREE.Color().setHSL(0.5 - persistence * 0.2, 1, 0.5);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            sizes[i] = persistence * 2;

            // Trail data (start at birth, go to death)
            trails.push({
                start: new THREE.Vector3(birth, birth, 0),
                end: new THREE.Vector3(birth, death, persistence)
            });
        }
        return { pts, colors, sizes, trails };
    }, [data]);

    useFrame((state) => {
        const t = state.clock.getElapsedTime();

        // Rotate the cloud slowly
        if (pointsRef.current) {
            pointsRef.current.rotation.y = t * 0.1;
            pointsRef.current.rotation.z = t * 0.05;
        }
        if (linesRef.current) {
            linesRef.current.rotation.y = t * 0.1;
            linesRef.current.rotation.z = t * 0.05;
        }
    });

    return (
        <group>
            {/* The Persistence Diagram Points */}
            <Points ref={pointsRef} positions={particles.pts} colors={particles.colors} sizes={particles.sizes}>
                <PointMaterial
                    transparent
                    vertexColors
                    size={0.15}
                    sizeAttenuation={true}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </Points>

            {/* Birth-Death Trails (The "Life" of the feature) */}
            <group ref={linesRef}>
                {particles.trails.map((trail, i) => (
                    <Line
                        key={i}
                        points={[trail.start, trail.end]}
                        color={i % 2 === 0 ? "#00ffff" : "#ff0055"} // Cyan/Red mix
                        lineWidth={1}
                        transparent
                        opacity={0.2}
                    />
                ))}
            </group>

            {/* Diagonal (y=x) reference line */}
            <Line
                points={[new THREE.Vector3(-2, -2, 0), new THREE.Vector3(2, 2, 0)]}
                color="#333"
                lineWidth={2}
            />
        </group>
    );
}
