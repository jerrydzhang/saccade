// A bounded pool: run `work` over `items` with at most `limit` in
// flight, in item order. The figure hydration used to await each
// figure's fetch before starting the next (t-36: 37 figures, ~1s of
// round trips one after another, m-547 (5)).
export async function inPool<T>(
  items: readonly T[],
  limit: number,
  work: (item: T, index: number) => Promise<void>,
): Promise<void> {
  let next = 0;
  const lane = async (): Promise<void> => {
    while (next < items.length) {
      const index = next++;
      await work(items[index]!, index);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, lane));
}
