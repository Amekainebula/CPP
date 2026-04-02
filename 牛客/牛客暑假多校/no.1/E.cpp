#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
void Murasame()
{
    int aa, bb;
    cin >> aa >> bb;
    i128 a = aa, b = bb;
    i128 n = max(a, b) * max(a, b) - min(a, b) * min(a, b);
    if (n == 3)
        cout << 1 << endl;
    else if (n == 5)
        cout << 2 << endl;
    int ans = 0;
    if (n % 4 == 3)
    {
        n++;
        ans = (n / 4 - 1) * 3 + 2 - 1;
    }
    else if (n % 4 == 1)
    {
        n--;
        ans = (n / 4 - 1) * 3 + 2 + 1;
    }
    else
    {
        ans = (n / 4 - 1) * 3 + 2;
    }
    cout << ans - 1 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}
// 3
// 5
// 7
// 8
// 9
// 11
// 12
// 13
// 15
// 16
// 17
// 19
// 20
// 21
// 23
// 24
// 25
// 27
// 28
// 29
// 31
// 32
// 33
// 35
// 36
// 37
// 39
// 40
// 41
// 43
// 44
// 45