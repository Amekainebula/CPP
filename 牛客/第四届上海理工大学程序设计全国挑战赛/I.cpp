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
    int x1, y1, r1, x2, y2, r2;
    cin >> x1 >> y1 >> r1 >> x2 >> y2 >> r2;
    int len1 = (r1 + r2) * (r1 + r2);
    int len2 = (r1 - r2) * (r1 - r2);
    int dis = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    if(x1==x2 && y1==y2&&r1==r2)
    {
        cout << -1 << endl;
        return;
    }
    if (dis > len1)
    {
        cout << 4 << endl;
    }
    else if (dis == len1)
    {
        cout << 3 << endl;
    }
    else if (dis < len1 && dis > len2)
    {
        cout << 2 << endl;
    }
    else if (dis == len2)
    {
        cout << 1 << endl;
    }
    else
    {
        cout << 0 << endl;
    }
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