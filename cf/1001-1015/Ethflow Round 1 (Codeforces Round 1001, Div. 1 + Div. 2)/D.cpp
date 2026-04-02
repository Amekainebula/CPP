#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
int v[400005];
vector<int> g[400005];
int ans;
int cnt;
void dfs(int u, int fa)
{
    for (int i = 0; i < sz(g[u]); i++)
    {
        int to = g[u][i];
        if (to == fa)
            continue;
        dfs(to, u);
    }

}
void solve()
{
    memset(g, 0, sizeof(g));
    int n;
    ans = 0;
    cnt = 0;
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> v[i];
    for (int i = 1; i < n; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u].pb(v);
    }
    dfs(1, 0);
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}