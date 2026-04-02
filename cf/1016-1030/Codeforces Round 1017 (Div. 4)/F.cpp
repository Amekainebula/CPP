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
#define vi vector<int>
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
void Murasame()
{
    int n, m, k;
    cin >> n >> m >> k;
    if (m % k == 0)
    {
        int t = 1;
        int now = 1;
        vi a(k + 1, 0);
        a[0] = k;
        ff(i, 1, k) a[i] = i;
        ff(i, 1, n)
        {
            ff(j, 1, m / k) ff(l, 1, k)
            {
                cout << a[l % k + (i % 2 == 0)] << " ";
            }
            cout << endl;
        }
    }
    else
    {
        int now = 1;
        ff(i, 1, n) ff(j, 1, m)
        {
            cout << now << " \n"[j == m];
            now++;
            if (now == k + 1)
                now = 1;
        }
    }
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