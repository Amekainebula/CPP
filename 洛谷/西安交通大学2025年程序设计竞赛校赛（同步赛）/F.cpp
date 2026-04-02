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
//
#define endl endl << flush
// #define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
int a1[100], a2[100];
int ask(int x)
{
    cout << "? " << x << endl;
    int res;
    cin >> res;
    return res;
}
void Murasame()
{
    int n;
    cin >> n;
    int now = 0;
    while (a2[now] < n)
    {
        now++;
        a1[now] = a2[now - 1] + 1;
        a2[now] = min(a1[now] * 4, n);
    }
    int l = 1, r = now;
    for (int i = 1; i <= 4; i++)
    {
        if (l > r)
            break;
        int mid = ((l + r + 1) >> 1);
        if (ask(a1[mid]) == 1)
            r = mid - 1;
        else
            l = mid;
    }
    cout << "! " << min(a2[l], a1[l] * 2) << endl;
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