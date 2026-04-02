#include <bits/stdc++.h>
// Finish Time: 2025/3/4 15:25:25 AC
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
void solve()
{
    int n;
    cin >> n;
    vector<int> a(n + 1);
    vector<int> ans(n + 1, 0);
    vector<int> in(n + 1, 0);
    vector<int> g[n + 1];
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
    {
        for (int j = i + a[i]; j <= n; j += a[i])
        {
            if (a[j] > a[i])
            {
                g[j].pb(i);
                in[i]++;
            }
        }
        for (int j = i - a[i]; j >= 1; j -= a[i])
        {
            if (a[j] > a[i])
            {
                g[j].pb(i);
                in[i]++;
            }
        }
    }
    queue<int> q;
    for (int i = 1; i <= n; i++)
        if (in[i] == 0)
            q.push(i);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        for (int v : g[u])
        {
            ans[v] += (!ans[u]);
            in[v]--;
            if (in[v] == 0)
                q.push(v);
        }
    }
    for (int i = 1; i <= n; i++)
        cout << "AB"[!ans[i]];
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