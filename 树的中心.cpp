#include <bits/stdc++.h>
#define ff(i, a, b) for (int i = (a); i <= (b); ++i)
#define ffg(i, a, b) for (int i = (a); i >= (b); --i)
#define endl '\n'
using namespace std;
const int N = 1e5 + 5;
int d1[N]; // 节点 𝑥 子树内的最长链。
int d2[N]; // 不与d1[𝑥]重叠最长链。
int up[N]; // 节点 𝑥 外的最长链。
int x, y, mn = LLONG_MAX, n;
struct edge
{
    int v, w;
};
vector<edge> g[N];
void dfs(int u, int fa) // 求取len1和len2
{
    for (auto &[v, w] : g[u])
    {
        if (v == fa)
            continue;
        dfs(v, u);
        if (d1[v] + w > d1[u])
        {
            d2[u] = d1[u];
            d1[u] = d1[v] + w;
        }
        else if (d1[v] + w > d2[u])
        {
            d2[u] = d1[v] + w;
        }
    }
}
void dfs2(int u, int fa) // 求取up
{
    for (auto &[v, w] : g[u])
    {
        if (v == fa)
            continue;
        up[v] = up[u] + w;
        if (d1[v] + w != d1[u]) // 如果自己子树里的最长链不在v子树里
        {
            up[v] = max(up[v], d1[u] + w);
        }
        else
        {
            up[v] = max(up[v], d2[u] + w);
        }
        dfs2(v, u);
    }
}

void get_center()
{
    dfs(1, 0);
    dfs2(1, 0);
    for (int i = 1; i <= n; i++)
    {
        if (max(d1[i], up[i]) < mn) // 找到了当前max(len1[x],up[x])最小点
        {
            mn = max(d1[i], up[i]);
            x = i, y = 0;
        }
        else if (max(d1[i], up[i]) == mn) // 另一个点
        {
            y = i;
        }
    }
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }
    get_center();
    cout << x << " " << y << endl;
    return 0;
}