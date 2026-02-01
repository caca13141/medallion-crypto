import React from 'react';
import {
    createColumnHelper,
    flexRender,
    getCoreRowModel,
    useReactTable,
    getSortedRowModel,
} from '@tanstack/react-table';

interface Position {
    symbol: string;
    side: 'BUY' | 'SELL';
    size: number;
    entry_price: number;
    mark_price: number;
    pnl: number;
    leverage: number;
}

const columnHelper = createColumnHelper<Position>();

const columns = [
    columnHelper.accessor('symbol', {
        header: 'SYMBOL',
        cell: info => <span className="font-bold text-[#111111]">{info.getValue()}</span>,
    }),
    columnHelper.accessor('side', {
        header: 'SIDE',
        cell: info => (
            <span className={`pill ${info.getValue() === 'BUY' ? 'pill-success' : 'pill-error'}`}>
                {info.getValue()}
            </span>
        ),
    }),
    columnHelper.accessor('size', {
        header: 'SIZE',
        cell: info => <span className="tabular-nums font-bold">{info.getValue().toFixed(4)}</span>,
    }),
    columnHelper.accessor('leverage', {
        header: 'LEV',
        cell: info => <span className="text-[10px] font-bold text-gray-400">{info.getValue().toFixed(1)}x</span>,
    }),
    columnHelper.accessor('entry_price', {
        header: 'ENTRY',
        cell: info => <span className="tabular-nums font-mono">${info.getValue().toLocaleString()}</span>,
    }),
    columnHelper.accessor('pnl', {
        header: 'PNL_USD',
        cell: info => {
            const val = info.getValue();
            return (
                <span className={`font-bold tabular-nums ${val > 0 ? 'text-[#0D9488]' : 'text-[#E11D48]'}`}>
                    {val > 0 ? '+' : ''}{val.toFixed(2)}
                </span>
            );
        },
    }),
];

export function PositionsTable({ positions }: { positions: Position[] }) {
    const table = useReactTable({
        data: positions,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        initialState: {
            sorting: [{ id: 'pnl', desc: true }],
        },
    });

    return (
        <div className="w-full h-full bg-white flex flex-col">
            <table className="enterprise-table">
                <thead>
                    {table.getHeaderGroups().map(headerGroup => (
                        <tr key={headerGroup.id}>
                            {headerGroup.headers.map(header => (
                                <th key={header.id}>
                                    {flexRender(header.column.columnDef.header, header.getContext())}
                                </th>
                            ))}
                        </tr>
                    ))}
                </thead>
                <tbody>
                    {table.getRowModel().rows.map(row => (
                        <tr key={row.id}>
                            {row.getVisibleCells().map(cell => (
                                <td key={cell.id}>
                                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
            {positions.length === 0 && (
                <div className="flex-1 flex flex-col items-center justify-center py-12">
                    <span className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">No Active Positions</span>
                </div>
            )}
        </div>
    );
}
