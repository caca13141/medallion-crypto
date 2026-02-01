import { useState } from 'react';
import { AlertOctagon } from 'lucide-react';

export function FlattenButton() {
    const [armed, setArmed] = useState(false);

    return (
        <button
            onClick={() => armed ? console.log("FLATTEN ALL") : setArmed(true)}
            onMouseLeave={() => setArmed(false)}
            className={clsx(
                "w-full h-full rounded border-2 flex flex-col items-center justify-center transition-all duration-200 uppercase font-black tracking-widest",
                armed
                    ? "bg-red-600 border-red-500 text-white animate-pulse scale-95"
                    : "bg-red-950/30 border-red-900/50 text-red-800 hover:bg-red-900/50 hover:text-red-500"
            )}
        >
            <AlertOctagon size={24} className="mb-1" />
            {armed ? "CONFIRM FLATTEN" : "PANIC FLATTEN"}
        </button>
    );
}

import { clsx } from 'clsx';
