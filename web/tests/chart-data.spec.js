import { test, expect } from '@playwright/test';
import { replayHistoryPoints } from '../src/charts.js';

for (const size of [13, 24]) {
  test('@software ' + `${size}点历史的末端始终在cutoff，不能画入预测区间`, () => {
    const input = Array.from({ length: size }, (_, i) => ({ wind_power: i }));
    const points = replayHistoryPoints(input);
    expect(points[0].x).toBe(-(size - 1) * 5);
    expect(points.at(-1).x).toBe(0);
    expect(points.every(point => point.x <= 0)).toBe(true);
    expect(points.map(point => point.y)).toEqual(input.map(row => row.wind_power));
    expect(points.slice(1).every((point, i) => point.x - points[i].x === 5)).toBe(true);
  });
}
