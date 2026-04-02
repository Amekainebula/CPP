#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mp make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define endl '\n'
using namespace std;
set<pii>f, g;
map<pii, int>ff, gg, sum;
void solve()
{
    int ans = 0;
    f.clear();
    g.clear();
    ff.clear();
    gg.clear();
    sum.clear();
    int n, m1, m2;
    cin >> n >> m1 >> m2;
    while (m1--)
    {
        int u, v;
        cin >> u >> v;
        f.insert(mp(u, v));
        ff[mp(u, v)]++;
        sum[mp(u, v)]++;
    }
    while (m2--)
    {
        int u, v;
        cin >> u >> v;
        g.insert(mp(u, v));
        gg[mp(u, v)]++;
        sum[mp(u, v)]++;
    }
    for (auto x : gg)
    {
        if (sum[x.fi] == 2)continue;
        if (f.find(x.fi) == f.end())
        {
            f.insert(x.fi);
            ans++;
        }
    }
    for (auto x : ff)
    {
        if (sum[x.fi] == 2)continue;
        if (g.find(x.fi) == g.end())
        {
            ans++;
        }
    }
    cout << ans << endl;
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