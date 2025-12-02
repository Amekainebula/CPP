#include <bits/stdc++.h>
#define ff(i, a, b) for (int i = (a); i <= (b); ++i)
#define ffg(i, a, b) for (int i = (a); i >= (b); --i)
#define endl '\n'
using namespace std;
const int N = 1e5 + 5;
int n;
vector<int> g[N];
class DfsW
{
public:
    int siz[N];            // 这个节点的「大小」（所有子树上节点数 + 该节点）
    int w[N];              // 这个节点的「重量」，即所有子树「大小」的最大值
    vector<int> centroids; // 重心集合

    void dfs(int u, int fa)
    {
        siz[u] = 1;
        w[u] = 0;
        for (int v : g[u])
        {
            if (v == fa)
                continue;
            dfs(v, u);
            siz[u] += siz[v];
            w[u] = max(w[u], siz[v]);
        }
        w[u] = max(w[u], n - siz[u]);
        if (w[u] <= n / 2)
        {
            centroids.push_back(u);
        }
    }
    vector<int> get_centroids()
    {
        dfs(1, 0);
        return centroids;
    }
};

class DpW
{
public:
    long long dp[N], ans[N], res = -1, res2 = -1;
    int siz[N];
    vector<int> centroids;
    void dfs(int u, int fa)
    {
        siz[u] = 1;
        dp[u] = 0;
        for (int v : g[u])
        {
            if (v == fa)
                continue;
            dfs(v, u);
            siz[u] += siz[v];
            dp[u] += dp[v] + siz[v];
        }
    }

    void dfs2(int u, int fa)
    {
        for (int v : g[u])
        {
            if (v == fa)
                continue;
            ans[v] = ans[u] - siz[v] + (n - siz[v]);
            dfs2(v, u);
        }
    }

    vector<int> get_centroids()
    {
        dfs(1, 0);
        ans[1] = dp[1];
        dfs2(1, 0);
        long long mini = LLONG_MAX;
        for (int i = 1; i <= n; i++)
        {
            if (ans[i] <= mini)
            {
                mini = ans[i];
                centroids.push_back(i);
            }
        }
        return centroids;
    }
};

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
        g[u].push_back(v);
        g[v].push_back(u);
    }
    DfsW dfsw;
    dfsw.get_centroids();
    DpW dpw;
    dpw.get_centroids();
    cout << dfsw.centroids.size() << endl;
    for (int i : dfsw.centroids)
    {
        cout << i << " ";
    }
    cout << endl;
    cout << dpw.centroids.size() << endl;
    for (int i : dpw.centroids)
    {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}