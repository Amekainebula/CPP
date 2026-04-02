
#include <bits/stdc++.h>

using namespace std;
// 匈牙利算法
// O(n*e)
const int N = 505;
int n, m, e;
int tim[N];   // 时间戳
int match[N]; // 匹配
vector<int> g[N];
bool dfs(int u, int t) // 当前节点, 时间戳
{
    if (tim[u] == t)
        return false;
    tim[u] = t;
    for (auto v : g[u])
    {
        if (!match[v] || dfs(match[v], t))
        {
            match[v] = u;
            return true;
        }
    }
    return false;
}
void solve1()
{
    cin >> n >> m >> e;
    for (int i = 1; i <= e; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
    }
    int ans = 0;
    for (int i = 1; i <= n; i++)
    {
        if (dfs(i, i))
            ans++;
    }
    cout << ans << '\n';
}

// Hopcroft–Karp算法
// O(sqrt(n)*e)
const int N = 1005; // 左部最大点数
const int M = 1005; // 右部最大点数

int n, m, e;
vector<int> g[N];
int matchX[N]; // 表示左部顶点 u 当前匹配到的右部顶点编号（如果未匹配则为 0）
int matchY[M]; // 表示右部顶点 v 当前匹配到的左部顶点编号（如果未匹配则为 0）
int distX[N];   
/*BFS 阶段构建层次图时，记录左部点 u 的“层次”或“距离”。
如果为 -1，表示该点在当前轮层次图中不可达。*/
bool visY[M]; // 记录右部顶点是否访问过。

// BFS：建立层次图，返回是否存在增广路径
bool bfs()
{
    queue<int> q;
    memset(distX, -1, sizeof(distX));

    for (int u = 1; u <= n; ++u)
    {
        if (matchX[u] == 0)
        { // 左部未匹配点入队
            distX[u] = 0;
            q.push(u);
        }
    }

    bool found = false;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        for (int v : g[u])
        {
            int w = matchY[v];
            if (w == 0)
            { // 到达右部未匹配点
                found = true;
            }
            else if (distX[w] == -1)
            {
                distX[w] = distX[u] + 1;
                q.push(w);
            }
        }
    }
    return found;
}

// DFS：在层次图中寻找增广路
bool dfs(int u)
{
    for (int v : g[u])
    {
        int w = matchY[v];
        if (w == 0 || (distX[w] == distX[u] + 1 && dfs(w)))
        {
            matchX[u] = v;
            matchY[v] = u;
            return true;
        }
    }
    distX[u] = -1; // 当前路径走不通，剪枝
    return false;
}
void solve2()
{
    cin >> n >> m >> e;
    for (int i = 0; i < e; ++i)
    {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
    }

    int ans = 0;
    while (bfs())
    {
        for (int u = 1; u <= n; ++u)
        {
            if (matchX[u] == 0 && dfs(u))
            {
                ans++;
            }
        }
    }

    cout << ans << '\n';
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    return 0;
}
