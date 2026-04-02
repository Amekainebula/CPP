# 题解：七影蝶的指引 (Shichieichou's Guidance) [easy version]

### 解题思路

#### 1. 预处理蝴蝶

首先计算每个节点 $u$ 包含的“有效蝴蝶”数量 $W[u]$。只有满足 $H - dep[u] \le t_u$ 的蝴蝶才有可能被指引。我们将每个节点上满足条件的蝴蝶数量求和，记为 $W[u]$。

#### 2. 定义状态

我们需要通过特殊能力来最大化路径上的 $W[u]$ 之和。

- **$uval[u]$**：从根节点 $1$ 到节点 $u$ 路径上所有 $W$ 的总和。

  $$uval[u] = W[u] + uval[parent(u)]$$

- **$dval[u]$**：从节点 $u$ 开始，向其子树方向延伸到某个叶子节点路径上的 $W$ 最大总和。

  $$dval[u] = W[u] + \max_{v \in children(u)} \{dval[v]\}$$

  （若 $u$ 是叶子，则 $dval[u] = W[u]$）

#### 3. 计算答案

如果不使用特殊能力，或者特殊能力在同一条垂直路径上使用，最大数量就是所有叶子节点中 $uval[L]$ 的最大值。

如果使用特殊能力进行“跨枝条”跳跃：

假设我们在深度 $d$ 处从节点 $u$ 跳到深度 $d-1$ 的节点 $v$。

- 收集到的蝴蝶总数为：**（从叶子到 $u$ 的最大蝴蝶数）+（从 $v$ 到根节点的蝴蝶数）**。
- 即：$dval[u] + uval[v]$。

为了最大化结果，对于每一个可能的跳跃深度 $d$（$1 \le d \le H$），我们取：

$$\text{Max\_at\_depth\_d} = \left( \max_{dep[u]=d} dval[u] \right) + \left( \max_{dep[v]=d-1} uval[v] \right)$$

最终答案即为所有层级跳跃方案与不跳跃方案中的最大值。



### 复杂度分析

- **时间复杂度：** $O(N + M)$。我们需要一次 DFS 预处理深度和父节点，两次遍历（自底向上和自顶向下）计算 $dval$ 和 $uval$。
- **空间复杂度：** $O(N + M)$，用于存储邻接表和节点属性。



### 参考代码(C++)

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

using i64 = long long;

const int N = 200005;
vector<int> adj[N];
vector<int> F[N];
int dep[N], fa[N];
i64 W[N], dval[N], uval[N];
int H;

void Dfs(int u, int p, int d)
{
    dep[u] = d;
    fa[u] = p;
    H = max(H, d);
    F[d].push_back(u);
    for (int v : adj[u])
    {
        if (v != p)
            Dfs(v, u, d + 1);
    }
}

void solve()
{
    int n, m;
    cin >> n >> m;

    H = 0;
    for (int i = 0; i <= n; ++i)
    {
        adj[i].clear();
        F[i].clear();
        W[i] = 0;
        dval[i] = 0;
        uval[i] = 0;
    }

    for (int i = 0; i < n - 1; ++i)
    {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    Dfs(1, 0, 0);

    for (int i = 0; i < m; ++i)
    {
        int a, t;
        cin >> a >> t;
        if (H - dep[a] <= t)
        {
            W[a] += 1;
        }
    }

    for (int d = H; d >= 0; --d)
    {
        for (int u : F[d])
        {
            i64 mx = 0;
            for (int v : adj[u])
            {
                if (v != fa[u])
                    mx = max(mx, dval[v]);
            }
            dval[u] = W[u] + mx;
        }
    }

    for (int d = 0; d <= H; ++d)
    {
        for (int u : F[d])
        {
            uval[u] = W[u] + uval[fa[u]];
        }
    }

    i64 ans = uval[F[H][0]];

    vector<i64> mxup(H + 1, 0);
    for (int d = 0; d <= H; ++d)
    {
        for (int u : F[d])
            mxup[d] = max(mxup[d], uval[u]);
    }

    for (int d = 1; d <= H; ++d)
    {
        i64 mxd = 0;
        for (int u : F[d])
            mxd = max(mxd, dval[u]);

        ans = max(ans, mxd + mxup[d - 1]);
    }

    cout << ans << endl;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int _T;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
```

