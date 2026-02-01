import { useState, useEffect } from 'react';
import { X } from 'lucide-react';

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    icon?: any;
    children: React.ReactNode;
    controls?: React.ReactNode;
}

export function ExpandedModal({ isOpen, onClose, title, icon: Icon, children, controls }: ModalProps) {
    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-50 bg-white/80 backdrop-blur-xl flex flex-col animate-in fade-in zoom-in-95 duration-200"
            onClick={onClose}
        >
            {/* Header */}
            <div className="h-16 border-b border-zinc-100 flex items-center justify-between px-6 bg-white/90" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center gap-3">
                    {Icon && <Icon size={18} className="text-zinc-400" />}
                    <h2 className="text-sm font-bold text-zinc-900 uppercase tracking-tight">{title}</h2>
                </div>

                {/* Controls */}
                <div className="flex items-center gap-4">
                    {controls}
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-zinc-100 rounded-full transition-colors"
                    >
                        <X size={18} className="text-zinc-400 hover:text-zinc-900" />
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 p-6 overflow-auto" onClick={(e) => e.stopPropagation()}>
                <div className="max-w-7xl mx-auto h-full">
                    {children}
                </div>
            </div>

            {/* Footer hint */}
            <div className="h-10 border-t border-zinc-100 flex items-center justify-center text-[10px] text-zinc-400 font-medium uppercase tracking-widest bg-zinc-50/50">
                Press <kbd className="mx-1.5 px-1.5 py-0.5 bg-white border border-zinc-200 rounded text-[9px] shadow-sm">ESC</kbd> to return to terminal
            </div>
        </div>
    );
}

// Hook for keyboard controls
export function useExpandedView(initialState = false) {
    const [isExpanded, setIsExpanded] = useState(initialState);

    const expand = () => setIsExpanded(true);
    const collapse = () => setIsExpanded(false);
    const toggle = () => setIsExpanded(!isExpanded);

    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') collapse();
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, []);

    return { isExpanded, expand, collapse, toggle };
}
