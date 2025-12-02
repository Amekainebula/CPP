#include <bits/stdc++.h>
using namespace std;
const int N = 5e5 + 100;

class LCA_BUL
{
public:
    vector<pair<int, int>> g[N];
    int dep[N], fa[N][31], cost[N][31];
    int n;

    void build(int _n)
    {
        memset(fa, 0, sizeof(fa));
        memset(cost, 0, sizeof(cost));
        memset(dep, 0, sizeof(dep));
        n = _n;
        for (int i = 1; i < n; i++)
        {
            int u, v;
            cin >> u >> v;
            g[u].push_back({v, 1});
            g[v].push_back({u, 1});
        }
    }

    void dfs(int root, int fno)
    {
        // 初始化：第 2^0 = 1 个祖先就是它的父亲节点，dep 也比父亲节点多 1。
        fa[root][0] = fno;
        dep[root] = dep[fa[root][0]] + 1;
        // 初始化：其他的祖先节点：第 2^i 的祖先节点是第 2^(i-1) 的祖先节点的第 2^(i-1) 的祖先节点。
        for (int i = 1; i < 31; i++)
        {
            int anc = fa[root][i - 1];
            fa[root][i] = fa[anc][i - 1];
            cost[root][i] = cost[anc][i - 1] + cost[root][i - 1];
        }
        // 遍历子节点，DFS处理。
        for (auto &[v, w] : g[root])
        {
            if (v == fno)
                continue;
            cost[v][0] = w;
            dfs(v, root);
        }
    }

    int lca(int x, int y)
    {
        // 令y比x深
        if (dep[x] > dep[y])
            swap(x, y);
        // 令 y 和 x 在一个深度。
        int temp = dep[y] - dep[x], ans = 0;
        for (int j = 0; temp; j++, temp >>= 1)
        {
            if (temp & 1)
                ans += cost[y][j], y = fa[y][j];
        }
        if (x == y)
            return ans;
        // 不然的话，找到第一个不是它们祖先的两个点。
        for (int j = 20; j >= 0 && y != x; --j)
        {
            if (fa[x][j] != fa[y][j])
            {
                ans += cost[x][j] + cost[y][j];
                x = fa[x][j];
                y = fa[y][j];
            }
        }
        // 返回结果。
        ans += cost[x][0] + cost[y][0];
        return ans;
    }
};
LCA_BUL lca;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int n, m, k;
    cin >> n >> m >> k;
    lca.build(n);
    lca.dep[k] = 1;
    lca.fa[k][0] = 0;
    lca.cost[k][0] = 0;
    lca.dfs(k, 0);
    for (int i = 1; i <= m; i++)
    {
        int x, y;
        cin >> x >> y;
        cout << lca.lca(x, y) << endl;
    }
    return 0;
}
