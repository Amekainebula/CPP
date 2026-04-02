#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
set<int> s;
map<int, int> mp;
void sp(int x, int y, int z)
{
    if ((x - y) / z * z == (x - y) && (x - y) / z >= 0)
        mp[(x - y) / z]++;
    if ((y - x) / z * z == (y - x) && (y - x) / z >= 0)
        mp[(y - x) / z]++;
    if ((x - z) / y * y == (x - z) && (x - z) / y >= 0)
        mp[(x - z) / y]++;
    if ((z - x) / y * y == (z - x) && (z - x) / y >= 0)
        mp[(z - x) / y]++;
    if ((y - z) / x * x == (y - z) && (y - z) / x >= 0)
        mp[(y - z) / x]++;
    if ((z - y) / x * x == (z - y) && (z - y) / x >= 0)
        mp[(z - y) / x]++;

    // s.clear();
}
void solve()
{
    mp.clear();
    int n;
    cin >> n;
    ff(i, 1, n)
    {
        int x, y, z;
        cin >> x >> y >> z;
        s.clear();
        sp(x, y, z);
    }
    for (auto it : mp)
    {
        if (it.se >= n)
        {
            cout << it.fi << endl;
            return;
        }
    }

    // cout<<endl;
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