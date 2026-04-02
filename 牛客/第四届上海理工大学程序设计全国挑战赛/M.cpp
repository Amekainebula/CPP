// Categories: Data Structures
#include <bits/stdc++.h>
#define i64 long long
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
    i64 a, b, c, d;
    cin >> a >> b >> c >> d;
    if (b == c && a != b && c != d && a != d)
        cout << "Yes" << endl;
    else
        cout << "No" << endl;
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