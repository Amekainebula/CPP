#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
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
int n;
vector<int> val(100005 + 1);
vector<bool> vis(100005 + 1, false);
vector<int> g[100005 + 1];
map<pii, int> w;
int ans = 0;
void dfs(int u, int sum)
{
    vis[u] = true;
    if (sum > val[u])
        return;
    ans++;
    for (int v : g[u])
    {
        if (!vis[v])
            dfs(v, max(w[{u, v}], sum + w[{u, v}]));
    }
}
void solve()
{
    cin >> n;
    for (int i = 1; i <= n; i++)
        cin >> val[i];
    for (int i = 2; i <= n; i++)
    {
        int x, y;
        cin >> x >> y;
        g[x].pb(i);
        g[i].pb(x);
        w[{x, i}] = y;
        w[{i, x}] = y;
    }
    dfs(1, 0);
    cout << n - ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}