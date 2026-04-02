#include <bits/stdc++.h>
#define ll long long
#define u64 unsigned long long
#define i128 __int128
#define d64 long double
#define ff(x, y, z) for (int(x) = (y); (x) < (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) > (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    ll n;
    cin >> n;
    vector<pii> v(n);
    ff(i, 0, n)
    {
        cin >> v[i].se;
        v[i].fi = i + 1;
    }
    auto cmp = [](pii a, pii b)
    { 
        if (a.se == b.se)
            return a.fi < b.fi;
        return a.se < b.se; 
    };
    sort(all(v), cmp);
    for (auto x : v)
        cout << x.fi << " ";
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