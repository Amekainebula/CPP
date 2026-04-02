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
const int mod = 998244353;

void solve()
{
    int maxx = -1e10;
    int ans = 0;
    int n;
    cin >> n;
    vector<int> g[n + 1];
    vector<int> hd[n + 1];
    vector<int> du[n + 1];
    map<int, int> mp;
    mp[1] = 0;
    for (int i = 2; i <= n; i++)
    {
        int x;
        cin >> x;
        g[x].pb(i);
        du[mp[x] + 1].pb(i);
        mp[i] = mp[x] + 1;
        maxx = max(maxx, mp[i]);
        hd[i].pb(x);
    }
    // dfs(1, 1);
    auto check = [&](int cnt)
    {
        for (auto x : du[cnt])
        {
            ans += du[cnt - 1].size() - hd[x].size();
            ans %= mod;
        }
    };
    for (int i = maxx; i > 1; i--)
        check(i);
    ans = (ans + g[1].size() + 1) % mod;
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