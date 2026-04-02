#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
    int x, y;
    cin >> x >> y;
    if (x == y || (x & y) == 0 || (x | y) == 0 || (x ^ y) == 0)
    {
        cout << 1 << endl;
        return;
    }
    int a = (x | y);
    int b = (x & y);
    int c = __gcd(x, y);
    int d = (x ^ y);
    if ((a & x) == 0 || (a & y) == 0 || (b & x) == 0 || (b & y) == 0 ||
        (c & x) == 0 || (c & y) == 0 || (d & x) == 0 || (d & y) == 0)
    {
        cout << 2 << endl;
        return;
    }
    if (a == x || a == y || b == x || b == y ||
        c == x || c == y || d == x || d == y)
    {
        cout << 2 << endl;
        return;
    }
    cout << 3 << endl;
    return;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}