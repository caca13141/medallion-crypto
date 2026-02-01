import React, { useState, useEffect } from 'react';

export default function SimpleApp() {
    const [debugInfo, setDebugInfo] = useState<any>({ lastTopic: 'waiting...', payloadKeys: [] });
    const [l3Data, setL3Data] = useState<any>(null);
    const [error, setError] = useState<string>('');

    useEffect(() => {
        try {
            const ws = new WebSocket('ws://localhost:3000/ws');
            ws.onopen = () => console.log('Connected');
            ws.onerror = (e) => setError('WS Error');
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.topic === 'marksman_pulse') {
                    const p = data.payload;
                    setDebugInfo({
                        lastTopic: 'marksman_pulse',
                        payloadKeys: Object.keys(p),
                        l3Size: p.l3_book ? 'YES' : 'NO'
                    });
                    if (p.l3_book) setL3Data(p.l3_book);
                } else {
                    setDebugInfo(prev => ({ ...prev, lastTopic: data.topic }));
                }
            };
            return () => ws.close();
        } catch (e: any) {
            setError(e.message);
        }
    }, []);

    return (
        <div className="p-10 text-white bg-[#111111] h-screen font-mono">
            <h1 className="text-4xl font-bold text-red-500 mb-4">CRASH DIAGNOSTICS MODE</h1>
            <div className="border border-green-500 p-6 rounded bg-black">
                <div className="mb-2">STATUS: <span className="text-green-400">RUNNING</span></div>
                <div className="mb-2">WS ERROR: <span className="text-red-400">{error || 'NONE'}</span></div>
                <div className="mb-2">LAST TOPIC: <span className="text-blue-400">{debugInfo.lastTopic}</span></div>
                <div className="mb-2">PAYLOAD KEYS: <span className="text-yellow-400">{debugInfo.payloadKeys.join(', ')}</span></div>
                <div className="mb-2">L3 DATA: <span className={`font-bold ${l3Data ? 'text-green-500' : 'text-red-500'}`}>{l3Data ? 'PRESENT' : 'NULL'}</span></div>
            </div>
            <div className="mt-8 text-gray-500 text-sm">
                If you see this, the React Mount is working. The failures are in the Components.
            </div>
        </div>
    );
}
