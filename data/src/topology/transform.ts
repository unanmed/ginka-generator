export function mirrorMapX(map: number[][]) {
    return map.map(v => [...v].reverse());
}

export function mirrorMapY(map: number[][]) {
    return [...map].reverse();
}

export function rotateMap(map: number[][]) {
    return [
        ...map[0].map((_, colIndex) => map.map(row => row[colIndex]))
    ].reverse();
}
