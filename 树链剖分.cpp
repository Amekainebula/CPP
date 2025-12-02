#include <bits/stdc++.h>

using namespace std;
const int N = 1e5 + 5;
vector<int> g[N];
int fa[N];  // fa[x]表示节点x的父亲
int dep[N]; // dep[x]表示节点x的深度
int siz[N]; // siz[x]表示以x为根的子树的结点数
int son[N]; // son[x]表示x的重儿子
int top[N]; // top[x]表示x所在的重链的顶部结点
int dfn[N]; // dfn[x]表示x的DFS序
int rnk[N]; // rnk[x]表示x的秩，有rnk[dfn[x]]=x
int tim = 0;
int NN, M, R, mod;

// 第一次dfs求fa,dep,siz,son
void dfs1(int u, int f)
{
    fa[u] = f, dep[u] = dep[f] + 1, siz[u] = 1;
    for (int v : g[u])
    {
        if (v == f)
            continue;
        dfs1(v, u);
        siz[u] += siz[v];
        if (siz[v] > siz[son[u]])
            son[u] = v;
    }
}

// 第二次dfs求top,dfn,rnk
void dfs2(int u, int ftop)
{
    top[u] = ftop, dfn[u] = ++tim, rnk[tim] = u;
    if (son[u])
        dfs2(son[u], ftop);
    for (int v : g[u])
    {
        if (v != son[u] && v != fa[u])
            dfs2(v, v);
    }
}

const int INF = LLONG_MAX >> 1;

struct SegTree
{
    struct Node
    {
        int l, r; // 当前区间范围
        int sum;  // 区间和 & 最大值（用于区间查询 / 单点修改）
        int num;  // 区域修改 & 单点查询（普通做法）
        int lazy; // lazy 标签（用于区间修改）
    } tree[N * 4];

    int a[N + 5];

    // --------------------------------------------
    // 建树：初始化区间信息
    // --------------------------------------------
    void build(int i, int l, int r)
    {
        tree[i].l = l, tree[i].r = r;
        tree[i].num = 0; // 默认无 lazy 区域修改

        if (l == r)
        { // 叶子节点赋值
            tree[i].sum = a[l];
            // tree[i].maxx = a[l];
            tree[i].num = a[l];
            return;
        }
        int mid = (l + r) >> 1;
        build(i << 1, l, mid);
        build(i << 1 | 1, mid + 1, r);

        tree[i].sum = (tree[i << 1].sum + tree[i << 1 | 1].sum) % mod;
        // tree[i].maxx = max(tree[i << 1].maxx, tree[i << 1 | 1].maxx);
    }

    // --------------------------------------------
    // lazy 下推
    // --------------------------------------------
    void push_down(int i)
    {
        if (tree[i].lazy)
        {
            tree[i << 1].lazy += tree[i].lazy;
            tree[i << 1].lazy %= mod;
            tree[i << 1 | 1].lazy += tree[i].lazy;
            tree[i << 1 | 1].lazy %= mod;

            int mid = (tree[i].l + tree[i].r) >> 1;
            tree[i << 1].sum += tree[i].lazy * (mid - tree[i << 1].l + 1);
            tree[i << 1].sum %= mod;

            tree[i << 1 | 1].sum += tree[i].lazy * (tree[i << 1 | 1].r - mid);
            tree[i << 1 | 1].sum %= mod;

            tree[i].lazy = 0;
        }
    }

    // --------------------------------------------
    // 带 lazy：区间加 k
    // --------------------------------------------
    void update_range_lazy(int i, int l, int r, int k)
    {
        if (tree[i].l >= l && tree[i].r <= r)
        {
            tree[i].sum += k * (tree[i].r - tree[i].l + 1);
            tree[i].sum %= mod;

            tree[i].lazy += k;
            tree[i].lazy %= mod;
            return;
        }
        push_down(i);

        if (tree[i << 1].r >= l)
            update_range_lazy(i << 1, l, r, k);
        if (tree[i << 1 | 1].l <= r)
            update_range_lazy(i << 1 | 1, l, r, k);

        tree[i].sum = (tree[i << 1].sum + tree[i << 1 | 1].sum) % mod;
    }

    // --------------------------------------------
    // 带 lazy：区间查询 sum
    // --------------------------------------------
    int query_sum_lazy(int i, int l, int r)
    {
        if (tree[i].l >= l && tree[i].r <= r)
            return tree[i].sum;
        if (tree[i].r < l || tree[i].l > r)
            return 0;

        push_down(i);
        int ans = 0;
        if (tree[i << 1].r >= l)
            ans += query_sum_lazy(i << 1, l, r), ans %= mod;
        if (tree[i << 1 | 1].l <= r)
            ans += query_sum_lazy(i << 1 | 1, l, r), ans %= mod;
        return ans;
    }
};
SegTree st;
// 常见应用

int qRrang(int x, int y)
{
    int ans = 0;
    while (top[x] != top[y])
    {
        if (dep[top[x]] < dep[top[y]])
            swap(x, y);
        ans = (ans + st.query_sum_lazy(1, dfn[top[x]], dfn[x])) % mod;
        x = fa[top[x]];
    }
    if (dep[x] > dep[y])
        swap(x, y);
    ans = (ans + st.query_sum_lazy(1, dfn[x], dfn[y])) % mod;
    return ans;
}

int qSon(int x)
{
    return st.query_sum_lazy(1, dfn[x], dfn[x] + siz[x] - 1) % mod;
}

void updRange(int x, int y, int k)
{
    k %= mod;
    while (top[x] != top[y])
    {
        if (dep[top[x]] < dep[top[y]])
            swap(x, y);
        st.update_range_lazy(1, dfn[top[x]], dfn[x], k);
        x = fa[top[x]];
    }
    if (dep[x] > dep[y])
        swap(x, y);
    st.update_range_lazy(1, dfn[x], dfn[y], k);
}

void updSon(int x, int k)
{
    st.update_range_lazy(1, dfn[x], dfn[x] + siz[x] - 1, k);
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> NN >> M >> R >> mod;
    for (int i = 1; i <= NN; i++)
    {
        cin >> st.a[i];
        st.a[i] %= mod;
    }
    for (int i = 1; i < NN; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs1(R, 0);
    dfs2(R, R);
    st.build(1, 1, NN);
    while (M--)
    {
        int k, x, y, z;
        cin >> k;
        if (k == 1)
        {
            cin >> x >> y >> z;
            updRange(x, y, z);
        }
        else if (k == 2)
        {
            cin >> x >> y;
            cout << qRrang(x, y) << endl;
        }
        else if (k == 3)
        {
            cin >> x >> y;
            updSon(x, y);
        }
        else
        {
            cin >> x;
            cout << qSon(x) << endl;
        }
    }
    return 0;
}