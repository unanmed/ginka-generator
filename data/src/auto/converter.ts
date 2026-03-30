import { sum } from 'lodash-es';
import { IAutoLabelConfig, ICodeRunResult } from './types';
import { CannotInOut, IMapTileConverter, ResourceType } from './types';

export class MapTileConverter implements IMapTileConverter {
    private readonly tower: ICodeRunResult;
    private readonly config: IAutoLabelConfig;
    private readonly noPassMap = new Map<number, boolean>();
    private readonly cannotInMap = new Map<number, number>();
    private readonly cannotOutMap = new Map<number, number>();
    private readonly labelMap = new Map<number, number>();
    private readonly resourceMap = new Map<number, Map<ResourceType, number>>();

    private readonly emptyTiles = new Set<number>([0]);
    private readonly doorTiles = new Set<number>();
    private readonly enemyTiles = new Set<number>();
    private readonly resourceTiles = new Set<number>();
    private readonly keyTiles = new Set<number>();
    private readonly itemTiles = new Set<number>();

    constructor(tower: ICodeRunResult, config: IAutoLabelConfig) {
        this.tower = tower;
        this.config = config;

        // 基于 maps 字典预计算各 tile 的分类与属性，避免重复解析。
        const tileMap = tower.map ?? {};
        for (const key of Object.keys(tileMap)) {
            const tile = Number(key);
            if (!Number.isFinite(tile)) continue;
            this.precomputeTile(tile);
        }
    }

    private static parseDirectionFlag(d: string | undefined): number {
        if (!d) return 0;
        switch (d) {
            case 'left':
                return CannotInOut.Left;
            case 'right':
                return CannotInOut.Right;
            case 'up':
            case 'top':
                return CannotInOut.Top;
            case 'down':
            case 'bottom':
                return CannotInOut.Bottom;
            default:
                return 0;
        }
    }

    private parseFlags(arr?: string[]): number {
        if (!arr || arr.length === 0) return 0;
        let result = 0;
        for (const d of arr) {
            result |= MapTileConverter.parseDirectionFlag(d);
        }
        return result;
    }

    private normalizeResource(values: Partial<Record<ResourceType, number>>) {
        const out = new Map<ResourceType, number>();
        for (const [k, v] of Object.entries(values)) {
            const n = Number(v);
            if (!Number.isFinite(n) || n <= 0) continue;
            out.set(Number(k) as ResourceType, n);
        }
        return out;
    }

    private evalItemEffectResource(
        itemEffect?: string
    ): Map<ResourceType, number> {
        if (!itemEffect) return new Map();

        const heroStatus = {
            hp: 0,
            atk: 0,
            def: 0,
            mdef: 0
        };
        const thisMap = { ratio: 1 };
        const values = this.tower.data?.values;
        if (!values) return new Map();

        const core = {
            values: new Proxy(values, {
                set() {
                    return true;
                }
            }),
            status: {
                hero: new Proxy(heroStatus, {
                    set(target, p: string, newValue) {
                        if (typeof newValue !== 'number') return true;
                        if (
                            p !== 'hp' &&
                            p !== 'atk' &&
                            p !== 'def' &&
                            p !== 'mdef'
                        ) {
                            return true;
                        }
                        target[p] = newValue;
                        return true;
                    }
                }),
                thisMap: new Proxy(thisMap, {
                    set() {
                        return true;
                    }
                })
            }
        };

        const log = console.log;
        console.log = () => {};
        try {
            eval(itemEffect);
        } catch (e) {
            console.log = log;
            return new Map();
        }
        console.log = log;

        return this.normalizeResource({
            [ResourceType.Hp]: heroStatus.hp,
            [ResourceType.Atk]: heroStatus.atk,
            [ResourceType.Def]: heroStatus.def,
            [ResourceType.Mdef]: heroStatus.mdef
        });
    }

    private precomputeTile(tile: number): void {
        if (this.labelMap.has(tile)) return;

        const labels = this.config.classes;
        const block = this.tower.map?.[tile];

        if (this.emptyTiles.has(tile)) {
            this.noPassMap.set(tile, false);
            this.cannotInMap.set(tile, 0);
            this.cannotOutMap.set(tile, 0);
            this.labelMap.set(tile, labels.empty);
            this.resourceMap.set(tile, new Map());
            return;
        }

        if (!block || tile === 17) {
            // 未知 tile 默认按墙处理
            this.noPassMap.set(tile, true);
            this.cannotInMap.set(tile, 0b1111);
            this.cannotOutMap.set(tile, 0b1111);
            this.labelMap.set(tile, labels.wall);
            this.resourceMap.set(tile, new Map());
            return;
        }

        const cannotIn = this.parseFlags(block.cannotIn);
        const cannotOut = this.parseFlags(block.cannotOut);
        this.cannotInMap.set(tile, cannotIn);
        this.cannotOutMap.set(tile, cannotOut);

        const blockId = block.id;
        const cls = block.cls;

        // 1. 钥匙资源识别（doorInfo.keys）
        let isKey = false;
        let keyCount = 0;
        if (block.doorInfo && block.doorInfo.keys) {
            const keys = block.doorInfo.keys;
            keyCount = sum(Object.values(keys));
            if (keyCount > 0) {
                isKey = true;
                this.keyTiles.add(tile);
            }
        }

        // 2. 道具资源识别（tools 且不是钥匙）
        let isItem = false;
        const item = this.tower.item?.[blockId];
        if (item?.cls === 'tools') {
            // 不是钥匙的 tools
            if (!isKey) {
                isItem = true;
                this.itemTiles.add(tile);
            }
        }

        const isDoor =
            block.doorInfo ||
            blockId.toLowerCase().endsWith('door') ||
            blockId === 'specialDoor';
        const isEnemy = cls === 'enemys' || cls === 'enemy48';

        let isResource = false;
        let resources = new Map<ResourceType, number>();
        if (isKey) {
            resources.set(ResourceType.Key, keyCount > 0 ? keyCount : 1);
            isResource = true;
        } else if (isItem) {
            resources.set(ResourceType.Item, 1);
            isResource = true;
        } else if (cls === 'items') {
            if (item?.cls === 'items') {
                resources = this.evalItemEffectResource(item?.itemEffect);
                isResource = resources.size > 0;
            } else if (item?.cls === 'equip') {
                if (item?.equip?.value) {
                    resources = this.normalizeResource({
                        [ResourceType.Hp]: item.equip.value.hp,
                        [ResourceType.Atk]: item.equip.value.atk,
                        [ResourceType.Def]: item.equip.value.def,
                        [ResourceType.Mdef]: item.equip.value.mdef
                    });
                    isResource = resources.size > 0;
                }
            }
        }

        const isEmpty =
            !isDoor && !isEnemy && !isResource && block.canPass === true;

        if (isDoor) {
            this.doorTiles.add(tile);
            this.noPassMap.set(tile, false);
            const label = labels.commonDoors[0];
            this.labelMap.set(tile, label);
        } else if (isEnemy) {
            this.enemyTiles.add(tile);
            this.noPassMap.set(tile, false);
            this.labelMap.set(tile, labels.enemies[0]);
        } else if (isResource) {
            this.resourceTiles.add(tile);
            this.noPassMap.set(tile, false);
            this.resourceMap.set(tile, resources);

            let label = labels.items[0] ?? labels.empty;
            const hp = resources.get(ResourceType.Hp) ?? 0;
            const atk = resources.get(ResourceType.Atk) ?? 0;
            const def = resources.get(ResourceType.Def) ?? 0;
            const mdef = resources.get(ResourceType.Mdef) ?? 0;
            const key = resources.get(ResourceType.Key) ?? 0;
            const item = resources.get(ResourceType.Item) ?? 0;
            const max = Math.max(hp, atk, def, mdef, key, item);
            if (max > 0) {
                if (max === hp) {
                    label = labels.potions[0] ?? label;
                } else if (max === atk) {
                    label = labels.redGems[0] ?? label;
                } else if (max === def) {
                    label = labels.blueGems[0] ?? label;
                } else if (max === mdef) {
                    label = labels.greenGems[0] ?? label;
                } else if (max === key) {
                    label = labels.keys[0] ?? label;
                } else if (max === item) {
                    label = labels.items[0] ?? label;
                }
            }
            this.labelMap.set(tile, label);
        } else if (isEmpty) {
            this.noPassMap.set(tile, false);
            this.labelMap.set(tile, labels.empty);
        } else if (block.canPass) {
            this.noPassMap.set(tile, false);
            this.labelMap.set(tile, labels.empty);
        } else {
            this.noPassMap.set(tile, true);
            this.labelMap.set(tile, labels.empty);
        }

        if (!this.resourceMap.has(tile)) {
            this.resourceMap.set(tile, new Map());
        }
    }

    getLabeledTile(tile: number): number {
        this.precomputeTile(tile);
        return this.labelMap.get(tile) ?? this.config.classes.wall;
    }

    isEmpty(tile: number): boolean {
        this.precomputeTile(tile);
        return this.labelMap.get(tile) === this.config.classes.empty;
    }

    isEntry(tile: number, x: number, y: number, floorId: string): boolean {
        const loc = `${x},${y}`;
        if (this.tower.main.floors[floorId].changeFloor[loc]) {
            return true;
        } else {
            return false;
        }
    }

    isDoor(tile: number): boolean {
        this.precomputeTile(tile);
        return this.doorTiles.has(tile);
    }

    isEnemy(tile: number): boolean {
        this.precomputeTile(tile);
        return this.enemyTiles.has(tile);
    }

    isResource(tile: number): boolean {
        this.precomputeTile(tile);
        return (
            this.resourceTiles.has(tile) ||
            this.keyTiles.has(tile) ||
            this.itemTiles.has(tile)
        );
    }

    getNoPass(tile: number, x: number, y: number): boolean {
        void x;
        void y;
        this.precomputeTile(tile);
        return this.noPassMap.get(tile) ?? true;
    }

    getCannotIn(tile: number, x: number, y: number): number {
        void x;
        void y;
        this.precomputeTile(tile);
        return this.cannotInMap.get(tile) ?? 0;
    }

    getCannotOut(tile: number, x: number, y: number): number {
        void x;
        void y;
        this.precomputeTile(tile);
        return this.cannotOutMap.get(tile) ?? 0;
    }

    getResource(tile: number, x: number, y: number): Map<ResourceType, number> {
        void x;
        void y;
        this.precomputeTile(tile);
        return new Map(this.resourceMap.get(tile) ?? []);
    }
}
